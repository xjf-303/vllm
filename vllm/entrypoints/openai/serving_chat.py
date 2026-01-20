# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Callable, Final, Optional, Union

import jinja2
import partial_json_parser
import regex as re
from fastapi import Request
from openai_harmony import Message as OpenAIMessage
from pydantic import TypeAdapter

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         ConversationMessage,
                                         get_history_tool_calls_cnt,
                                         make_tool_call_id)
from vllm.entrypoints.harmony_utils import (
    get_developer_message, get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant, get_system_message, parse_chat_input,
    parse_chat_output, render_for_completion)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb, ChatCompletionLogProbs,
    ChatCompletionLogProbsContent, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaFunctionCall, DeltaMessage,
    DeltaToolCall, ErrorResponse, FunctionCall, FunctionDefinition,
    PromptTokenUsageInfo, RequestResponseMetadata, ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    clamp_prompt_logprobs)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
    MistralToolCall)
from vllm.entrypoints.utils import get_max_tokens
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.transformers_utils.tokenizers import (maybe_serialize_tool_calls,
                                                truncate_tool_call_ids,
                                                validate_request_params)
from vllm.utils import as_list, random_uuid

# YoutuVL two-stage decoding imports
try:
    from vllm.model_executor.models.youtuvl import (
        YoutuVLLayoutParser, LayoutElement
    )
    HAS_YOUTUVL = True
except ImportError:
    HAS_YOUTUVL = False
    YoutuVLLayoutParser = None
    LayoutElement = None

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: Optional[str] = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids,
                         enable_force_include_usage=enable_force_include_usage,
                         log_error_stack=log_error_stack)

        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.enable_log_outputs = enable_log_outputs

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                "\"auto\" tool choice has been enabled please note that while"
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored.")

        self.reasoning_parser: Optional[Callable[[AnyTokenizer],
                                                 ReasoningParser]] = None
        if reasoning_parser:
            try:
                self.reasoning_parser = (
                    ReasoningParserManager.get_reasoning_parser(
                        reasoning_parser))
                assert self.reasoning_parser is not None
            except Exception as e:
                raise TypeError(
                    f"{reasoning_parser=} has not been registered") from e
        self.tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None
        if self.enable_auto_tools:
            try:
                if (tool_parser == "pythonic" and
                        model_config.model.startswith("meta-llama/Llama-3.2")):
                    logger.warning(
                        "Llama3.2 models may struggle to emit valid pythonic"
                        " tool calls")
                self.tool_parser = ToolParserManager.get_tool_parser(
                    tool_parser)
            except Exception as e:
                raise TypeError("Error: --enable-auto-tool-choice requires "
                                f"tool_parser:'{tool_parser}' which has not "
                                "been registered") from e
        self.exclude_tools_when_tool_choice_none = (
            exclude_tools_when_tool_choice_none)

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default chat sampling params from %s: %s",
                        source, self.default_sampling_params)
        if self.model_config.hf_config.model_type == 'kimi_k2':
            self.tool_call_id_type = 'kimi_k2'
        else:
            self.tool_call_id_type = 'random'

        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions())

        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse,
               ErrorResponse]:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # ========== YoutuVL Mode Detection ==========
        youtuvl_mode = getattr(request, 'youtuvl_mode', None)
        if youtuvl_mode and youtuvl_mode != "chat":
            if not HAS_YOUTUVL:
                return self.create_error_response(
                    "YoutuVL two-stage decoding is not available. "
                    "Please ensure youtuvl model is properly installed."
                )
            return await self._handle_youtuvl_request(request, raw_request)
        # ============================================

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            lora_request = self._maybe_get_adapters(
                request, supports_default_mm_loras=True)

            model_name = self.models.model_name(lora_request)

            tokenizer = await self.engine_client.get_tokenizer()

            tool_parser = self.tool_parser

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)
                truncate_tool_call_ids(request)
                validate_request_params(request)

            if (request.tool_choice == "auto" and
                    not (self.enable_auto_tools and tool_parser is not None)
                    and not isinstance(tokenizer, MistralTokenizer)
                    and not self.use_harmony):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    "\"auto\" tool choice requires "
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )

            if (request.tools is None
                    or (request.tool_choice == "none"
                        and self.exclude_tools_when_tool_choice_none)):
                tool_dicts = None
            else:
                tool_dicts = [tool.model_dump() for tool in request.tools]

            if not self.use_harmony:
                # Common case.
                request_chat_template = request.chat_template
                chat_template_kwargs = request.chat_template_kwargs
                if not self.trust_request_chat_template and (
                        request_chat_template is not None or
                    (chat_template_kwargs and
                     chat_template_kwargs.get("chat_template") is not None)):
                    return self.create_error_response(
                        "Chat template is passed with request, but "
                        "--trust-request-chat-template is not set. "
                        "Refused request with untrusted chat template.")
                (
                    conversation,
                    request_prompts,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request_chat_template or self.chat_template,
                    chat_template_content_format=self.
                    chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    tool_dicts=tool_dicts,
                    documents=request.documents,
                    chat_template_kwargs=request.chat_template_kwargs,
                    tool_parser=tool_parser,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                # For GPT-OSS.
                (
                    conversation,
                    request_prompts,
                    engine_prompts,
                ) = self._make_request_with_harmony(request)
        except (ValueError, TypeError, RuntimeError,
                jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_id = "chatcmpl-" \
                     f"{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: Union[SamplingParams, BeamSearchParams]

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=len(engine_prompt["prompt_token_ids"]),
                    default_sampling_params=self.default_sampling_params)

                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens, self.model_config.logits_processor_pattern,
                        self.default_sampling_params)

                self._log_inputs(request_id,
                                 request_prompts[i],
                                 params=sampling_params,
                                 lora_request=lora_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                else:
                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        result_generator, = generators

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                enable_force_include_usage=self.enable_force_include_usage)

        try:
            return await self.chat_completion_full_generator(
                request, result_generator, request_id, model_name,
                conversation, tokenizer, request_metadata)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    @staticmethod
    def _bracket_level(s: str, opening='{', closing='}') -> int:
        """
        Calculate the current level of nested brackets in a given string.
        """
        level = 0
        for char in s:
            if char == opening:
                level += 1
            elif char == closing:
                level -= 1
        return level

    @staticmethod
    def _filter_delta_text(delta_text: str,
                           previous_text: str) -> tuple[str, bool]:
        # remove last '},' of the tool definition stemming from the
        # "name"/"parameters" outer object or closing ']' of the tool list
        # count occurrences of opening and closing curly braces and
        # once level 0 is reached stop outputting text
        # if 0 is reached while parsing the delta_text we know the current
        # tool will finish in this current iteration
        bracket_level = OpenAIServingChat._bracket_level(previous_text)
        updated_delta, passed_zero = "", False
        for c in delta_text:
            if c == '{':
                bracket_level += 1
                passed_zero = bracket_level == 0
            elif c == '}':
                bracket_level -= 1
                passed_zero = bracket_level == 0

            if bracket_level != 0:
                updated_delta += c
            else:
                # if a comma is reached at level 0 we can stop
                if c == ',':
                    break
        return updated_delta, passed_zero

    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: Optional[str],
        delta_text: str,
        function_name_returned: bool,
        tool_call_idx: Optional[int] = None
    ) -> tuple[Optional[DeltaMessage], bool]:
        if current_text is None or current_text == "":
            # if the current text is empty, we cannot parse it
            return None, function_name_returned
        try:
            obj = partial_json_parser.loads(current_text)
        except partial_json_parser.core.exceptions.MalformedJSON:
            logger.debug('not enough tokens to parse into JSON yet')
            obj = None

        # check if the current text is a valid array
        # containing a partial tool calling object
        # if not repeat
        if obj is None or not isinstance(obj, list) or not len(obj) > 0:
            function_name_returned = False
            delta_message = None
        else:
            _, finishes_previous_tool = OpenAIServingChat._filter_delta_text(
                delta_text, previous_text)
            # take the last tool call from the generated list
            current_tool_call = obj[-1]

            # once parameters have been generated the name is complete as well
            if not finishes_previous_tool and ("name" not in current_tool_call
                                               or "parameters"
                                               not in current_tool_call):
                function_name_returned = False
                delta_message = None
            else:
                if not function_name_returned:
                    # get partly generated arguments from the latest tool call
                    param_match = re.search(r'.*"parameters":\s*(.*)',
                                            current_text, re.DOTALL)
                    arguments = param_match.group(1) if param_match else ""
                    arguments, _ = OpenAIServingChat._filter_delta_text(
                        arguments, previous_text)

                    # if this iteration finishes a previous tool call but a
                    # new incomplete tool is already generated, take the
                    # previous from the list
                    if (finishes_previous_tool
                            and "parameters" not in current_tool_call):
                        current_tool_call = obj[-2]

                    function_name_returned = True
                    tool_call_id = make_tool_call_id(
                        id_type=self.tool_call_id_type,
                        func_name=current_tool_call["name"],
                        idx=tool_call_idx)
                    delta_message = DeltaMessage(tool_calls=[
                        DeltaToolCall(id=tool_call_id,
                                      function=DeltaFunctionCall(
                                          name=current_tool_call["name"],
                                          arguments=arguments),
                                      index=len(obj) - 1,
                                      type="function")
                    ])

                else:
                    delta_text, _ = OpenAIServingChat._filter_delta_text(
                        delta_text, previous_text)

                    if delta_text != "":
                        delta_message = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                function=DeltaFunctionCall(
                                    # OpenAI API returns None
                                    # instead of name every time
                                    name=None,
                                    arguments=delta_text),
                                index=len(obj) - 1)
                        ])
                    else:
                        delta_message = None

        return delta_message, function_name_returned

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        enable_force_include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant()
                for _ in range(num_choices)
            ]
            harmony_tools_streamed = [False] * num_choices
        tools_streamed = [False] * num_choices

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request))

        all_previous_token_ids: Optional[list[list[int]]]
        function_name_returned = [False] * num_choices
        if self.tool_call_id_type == 'kimi_k2':
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto or self.reasoning_parser:
            # These are only required in "auto" tool choice case
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        elif request.tool_choice == "required":
            all_previous_token_ids = None
        else:
            all_previous_token_ids = None

        try:
            if self.reasoning_parser:
                reasoning_parser = self.reasoning_parser(tokenizer)
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return
        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and self.tool_parser:
                tool_parsers: list[Optional[ToolParser]] = [
                    self.tool_parser(tokenizer)
                ] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        if stream_options:
            include_usage = stream_options.include_usage \
                            or enable_force_include_usage
            include_continuous_usage = include_usage and \
                                       stream_options.continuous_usage_stats
        else:
            include_usage, include_continuous_usage = False, False

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None)

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(res.prompt_token_ids
                                              if request.return_token_ids else
                                              None))

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens)

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: Union[str, list[dict[str, str]]] = ""
                        if conversation and "content" in conversation[
                                -1] and conversation[-1].get("role") == role:
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        logprobs=None,
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name)
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens)

                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = tool_parsers[i]

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.
                            return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    if self.use_harmony:
                        harmony_parser = harmony_parsers[i]
                        prev_recipient = harmony_parser.current_recipient
                        for token_id in output.token_ids:
                            harmony_parser.process(token_id)
                        cur_channel = harmony_parser.current_channel
                        cur_recipient = harmony_parser.current_recipient
                        delta_text = harmony_parser.last_content_delta or ""
                    else:
                        delta_text = output.text

                    if not delta_text and not output.token_ids and \
                        not previous_num_tokens[i]:
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: Optional[DeltaMessage]

                    # just update previous_texts and previous_token_ids
                    if tool_choice_auto or self.reasoning_parser:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        # avoid the None + list error.
                        if previous_token_ids:
                            current_token_ids = previous_token_ids + as_list(
                                output.token_ids)
                        else:
                            current_token_ids = as_list(output.token_ids)

                    if self.use_harmony:
                        if cur_channel == "final":
                            delta_message = DeltaMessage(content=delta_text)
                        elif cur_channel == "analysis":
                            if request.include_reasoning:
                                delta_message = DeltaMessage(
                                    reasoning_content=delta_text)
                            else:
                                delta_message = None
                        elif (cur_channel == "commentary" and cur_recipient
                              and cur_recipient.startswith("functions.")):
                            # Count completed tool calls to determine index
                            base_index = 0
                            for msg in harmony_parser.messages:
                                if (msg.channel == "commentary"
                                        and msg.recipient
                                        and msg.recipient.startswith(
                                            "functions.")):
                                    base_index += 1

                            if prev_recipient != cur_recipient:
                                tool_name = cur_recipient.split(
                                    "functions.", 1)[1]
                                delta_message = DeltaMessage(tool_calls=[
                                    DeltaToolCall(
                                        id=make_tool_call_id(),
                                        type="function",
                                        function=DeltaFunctionCall(
                                            name=tool_name,
                                            arguments="",
                                        ),
                                        index=base_index,
                                    )
                                ])
                            elif delta_text:
                                delta_message = DeltaMessage(tool_calls=[
                                    DeltaToolCall(
                                        index=base_index,
                                        function=DeltaFunctionCall(
                                            arguments=delta_text),
                                    )
                                ])
                            else:
                                delta_message = None

                            if delta_message is not None:
                                harmony_tools_streamed[i] = True
                        else:
                            delta_message = None
                    # handle streaming deltas for tools with named tool_choice
                    elif tool_choice_function_name:
                        if (self.reasoning_parser and not reasoning_end_arr[i]
                                and not reasoning_parser.is_reasoning_end(
                                    previous_token_ids)):
                            assert reasoning_parser is not None
                            delta_message = (
                                reasoning_parser.
                                extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                ))
                            # When encountering think end id in delta_token_ids
                            # or think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            # Only keep 'content', remove 'reasoning_content'.
                            if reasoning_parser.is_reasoning_end(
                                    as_list(output.token_ids)) or (
                                        res.prompt_token_ids
                                        and reasoning_parser.is_reasoning_end(
                                            res.prompt_token_ids)):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    # This need to be added to next `delta_text`
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if self.reasoning_parser:
                                delta_text = previous_text + delta_text
                                current_text = ""

                            if function_name_returned[i]:
                                delta_tool_call = DeltaToolCall(
                                    function=DeltaFunctionCall(
                                        arguments=delta_text),
                                    index=i)
                            else:
                                delta_tool_call = DeltaToolCall(
                                    id=make_tool_call_id(),
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=tool_choice_function_name,
                                        arguments=delta_text),
                                    index=i)
                                function_name_returned[i] = True

                            delta_message = DeltaMessage(tool_calls=[
                                delta_tool_call,
                            ])
                            tools_streamed[i] = True

                    elif request.tool_choice == "required":
                        assert previous_texts is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]

                        if self.reasoning_parser:
                            _, content = \
                                reasoning_parser.extract_reasoning_content(
                                    current_text,
                                    request
                                )
                        else:
                            content = current_text
                        delta_message, function_name_returned[i] = (
                            self.extract_tool_call_required_streaming(
                                previous_text=previous_text,
                                current_text=content,
                                delta_text=delta_text,
                                function_name_returned=fn_name_returned,
                                tool_call_idx=history_tool_call_cnt))
                        if (delta_message and delta_message.tool_calls and
                                delta_message.tool_calls[0].id is not None):
                            history_tool_call_cnt += 1
                            tools_streamed[i] = True

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser
                    elif tool_choice_auto and self.reasoning_parser:
                        assert tool_parser is not None
                        assert reasoning_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        if not reasoning_end_arr[i]:
                            delta_message = (
                                reasoning_parser.
                                extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output_token_ids,
                                ))
                            # When encountering think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if res.prompt_token_ids and \
                                reasoning_parser.is_reasoning_end(
                                    res.prompt_token_ids):
                                reasoning_end_arr[i] = True
                                current_token_ids = output_token_ids
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                            # When encountering think end id in delta_token_ids,
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if reasoning_parser.is_reasoning_end(
                                    output_token_ids):
                                reasoning_end_arr[i] = True
                                current_token_ids =  \
                                    reasoning_parser.extract_content_ids(
                                        output_token_ids)
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""

                        # handle tool calls only after reasoning is done,
                        else:
                            delta_token_ids = output_token_ids
                            # First time to tool call,
                            # add the remaining text and token ids
                            # to delta from previous
                            if not added_content_delta_arr[i]:
                                added_content_delta_arr[i] = True
                                previous_text = ""
                                previous_token_ids = []
                                delta_text = current_text
                                delta_token_ids = current_token_ids

                            delta_message = (
                                tool_parser.extract_tool_calls_streaming(
                                    previous_text=previous_text,
                                    current_text=current_text,
                                    delta_text=delta_text,
                                    previous_token_ids=previous_token_ids,
                                    current_token_ids=current_token_ids,
                                    delta_token_ids=delta_token_ids,
                                    request=request))
                            if delta_message and delta_message.tool_calls:
                                tools_streamed[i] = True
                    # when only tool calls
                    elif tool_choice_auto:
                        assert tool_parser is not None
                        delta_message = (
                            tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=output.token_ids,
                                request=request))
                        if delta_message and delta_message.tool_calls:
                            tools_streamed[i] = True

                    # when only reasoning
                    elif self.reasoning_parser:
                        delta_message = (reasoning_parser.
                                         extract_reasoning_content_streaming(
                                             previous_text,
                                             current_text,
                                             delta_text,
                                             previous_token_ids,
                                             current_token_ids,
                                             output.token_ids,
                                         ))
                    # handle streaming just a content delta
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if ((tool_choice_auto or self.reasoning_parser)
                            and not self.use_harmony):
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids
                    else:
                        # Update for comprehensive logging even in simple case
                        assert previous_texts is not None
                        previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        if output.finish_reason is None:
                            continue
                        else:
                            delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content = ""
                        if delta_message.content:
                            delta_content = delta_message.content
                        elif delta_message.tool_calls:
                            delta_content = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments)

                        if delta_content:
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(as_list(output.token_ids)
                                       if request.return_token_ids else None))

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using structured outputs
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(
                                tool_parser.prev_tool_call_arr) > 0
                            index = len(tool_parser.prev_tool_call_arr
                                        ) - 1 if auto_tools_called else 0
                        else:
                            index = 0

                        if self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output) and tool_parser:
                            latest_delta_len = 0
                            if ((isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall)) and isinstance(
                                        delta_message.tool_calls[0].function.
                                        arguments, str)):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.
                                    arguments)

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON
                            expected_call = json.dumps(
                                tool_parser.prev_tool_call_arr[index].get(
                                    "arguments", {}),
                                ensure_ascii=False)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[
                                index]
                            if (latest_delta_len > 0):
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(
                                actual_call, "", 1)
                            # set that as a delta message
                            delta_message = DeltaMessage(tool_calls=[
                                DeltaToolCall(index=index,
                                              function=DeltaFunctionCall(
                                                  arguments=remaining_call).
                                              model_dump(exclude_none=True))
                            ])

                        # Send the finish response for each request.n only once
                        if auto_tools_called or tools_streamed[i] or (
                                self.use_harmony
                                and harmony_tools_streamed[i]):
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = output.finish_reason \
                                if output.finish_reason else "stop"
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(as_list(output.token_ids)
                                       if request.return_token_ids else None))

                        finish_reason_sent[i] = True

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=num_prompt_tokens +
                                        completion_tokens)
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens)

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts) else
                        f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=
                        None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        choices: list[ChatCompletionResponseChoice] = []
        if self.tool_call_id_type == 'kimi_k2':
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs
            tool_call_info = None

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if self.use_harmony:
                reasoning_content, content, _ = parse_chat_output(token_ids)
                if not request.include_reasoning:
                    reasoning_content = None

                if self.tool_parser is not None:
                    tool_parser = self.tool_parser(tokenizer)
                    # NOTE: We use token_ids for openai tool parser
                    tool_call_info = tool_parser.extract_tool_calls(
                        "",
                        request=request,
                        token_ids=token_ids,  # type: ignore
                    )
                    content = tool_call_info.content
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=content,
                        tool_calls=tool_call_info.tool_calls,
                    )
                else:
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=content,
                    )

                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    logprobs=logprobs,
                    finish_reason="tool_calls" if
                    (tool_call_info is not None
                     and tool_call_info.tools_called) else
                    output.finish_reason if output.finish_reason else "stop",
                    stop_reason=output.stop_reason,
                )
                choices.append(choice_data)
                continue

            if self.reasoning_parser:
                try:
                    reasoning_parser = self.reasoning_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning_content, content = (
                    reasoning_parser.extract_reasoning_content(
                        output.text, request=request))
                if not request.include_reasoning:
                    reasoning_content = None
            else:
                reasoning_content = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if (not self.enable_auto_tools or not self.tool_parser) and \
                (not isinstance(request.tool_choice,
                                ChatCompletionNamedToolChoiceParam
                                ) and request.tool_choice != "required"):
                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=content)

            # if the request uses tools and specified a tool choice
            elif request.tool_choice and type(
                    request.tool_choice) is ChatCompletionNamedToolChoiceParam:

                tool_call_class = MistralToolCall if isinstance(
                    tokenizer, MistralTokenizer) else ToolCall
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content="",
                    tool_calls=[
                        tool_call_class(function=FunctionCall(
                            name=request.tool_choice.function.name,
                            arguments=content,
                        ))
                    ],
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class = MistralToolCall if isinstance(
                    tokenizer, MistralTokenizer) else ToolCall

                # the fields of FunctionDefinition are a superset of the
                # tool call outputs and can be used for parsing
                assert content is not None
                tool_calls = TypeAdapter(
                    list[FunctionDefinition]).validate_json(content)
                tool_call_ids = []
                for tool_call in tool_calls:
                    tool_call_ids.append(
                        make_tool_call_id(id_type=self.tool_call_id_type,
                                          func_name=tool_call.name,
                                          idx=history_tool_call_cnt))
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        tool_call_class(id=tool_call_ids[i],
                                        function=FunctionCall(
                                            name=tool_call.name,
                                            arguments=json.dumps(
                                                tool_call.parameters,
                                                ensure_ascii=False)))
                        for i, tool_call in enumerate(tool_calls)
                    ],
                    reasoning_content=reasoning_content)

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":

                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=content)

            # handle when there are tools and tool choice is auto
            elif request.tools and (
                    request.tool_choice == "auto"
                    or request.tool_choice is None) and self.enable_auto_tools \
                    and self.tool_parser:

                try:
                    tool_parser = self.tool_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in tool parser creation.")
                    return self.create_error_response(str(e))

                tool_call_info = tool_parser.extract_tool_calls(
                    content if content is not None else "", request=request)
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message = ChatMessage(role=role,
                                          reasoning_content=reasoning_content,
                                          content=tool_call_info.content,
                                          tool_calls=tool_call_info.tool_calls)

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    ret_content = content

                    # try to use content return from tool parser first,
                    # tool parser may do some modify for the content.
                    if (tool_call_info.content
                            and len(tool_call_info.content) > 0):
                        ret_content = tool_call_info.content
                    message = ChatMessage(role=role,
                                          reasoning_content=reasoning_content,
                                          content=ret_content)

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion.")
                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=content)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls" if auto_tools_called else
                output.finish_reason if output.finish_reason else "stop",
                stop_reason=output.stop_reason,
                token_ids=(as_list(output.token_ids)
                           if request.return_token_ids else None),
            )

            choices.append(choice_data)

        if request.echo:
            last_msg_content: Union[str, list[dict[str, str]]] = ""
            if (conversation and "content" in conversation[-1]
                    and conversation[-1].get("role") == role):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg['text']
                                             for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content
                                                   or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                          completion_tokens=num_generated_tokens,
                          total_tokens=num_prompt_tokens +
                          num_generated_tokens)
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens)

        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(final_res.prompt_token_ids
                              if request.return_token_ids else None),
            kv_transfer_params=final_res.kv_transfer_params,
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:
                        if hasattr(tc.function, "name") and hasattr(
                                tc.function, "arguments"):
                            tool_call_descriptions.append(
                                f"{tc.function.name}({tc.function.arguments})")
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[
                            choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _get_top_logprobs(
            self, logprobs: dict[int, Logprob], top_logprobs: Optional[int],
            tokenizer: AnyTokenizer,
            should_return_as_token_id: bool) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(token := self._get_decoded_token(
                    p[1],
                    p[0],
                    tokenizer,
                    return_as_token_id=should_return_as_token_id,
                )),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            ) for i, p in enumerate(logprobs.items())
            if top_logprobs and i < top_logprobs
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[dict[int, Logprob]]],
        tokenizer: AnyTokenizer,
        num_output_top_logprobs: Optional[int] = None,
        return_as_token_id: Optional[bool] = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = return_as_token_id if \
            return_as_token_id is not None else self.return_tokens_as_token_ids
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(
                    token_id) is None:
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"
                else:
                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    ))
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=None if step_decoded is None else list(
                            step_decoded.encode("utf-8", errors="replace")),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs, num_output_top_logprobs,
                            tokenizer, should_return_as_token_id),
                    ))

        return ChatCompletionLogProbs(content=logprobs_content)

    def _should_stream_with_auto_tool_parsing(self,
                                              request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (request.tools and self.tool_parser and self.enable_auto_tools
                and request.tool_choice in ['auto', None])

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: Optional[DeltaMessage],
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        # yapf: disable
        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools and self.tool_parser and delta_message
            and delta_message.tool_calls and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )

    def _make_request_with_harmony(
        self,
        request: ChatCompletionRequest,
    ):
        messages: list[OpenAIMessage] = []

        # Add system message.
        # NOTE: In Chat Completion API, browsing is enabled by default
        # if the model supports it. TODO: Support browsing.
        assert not self.supports_browsing
        assert not self.supports_code_interpreter
        sys_msg = get_system_message(
            reasoning_effort=request.reasoning_effort,
            browser_description=None,
            python_description=None)
        messages.append(sys_msg)

        # Add developer message.
        dev_msg = get_developer_message(tools=request.tools)
        messages.append(dev_msg)

        # Add user message.
        for chat_msg in request.messages:
            messages.extend(parse_chat_input(chat_msg))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_token_ids)

        # Add cache_salt if provided in the request
        if request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return messages, [prompt_token_ids], [engine_prompt]

    # ========== YoutuVL Two-Stage Decoding Methods ==========

    # Debug flag for timing statistics
    _YOUTUVL_DEBUG = os.environ.get("YOUTUVL_DEBUG", "0") == "1"

    async def _handle_youtuvl_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Handle YoutuVL document parsing request."""
        mode = getattr(request, 'youtuvl_mode', 'chat')

        try:
            if mode == "layout":
                return await self._youtuvl_layout_detect(request, raw_request)
            elif mode == "ocr":
                return await self._youtuvl_ocr_recognize(request, raw_request)
            elif mode == "document":
                return await self._youtuvl_document_parse(request, raw_request)
            else:
                return self.create_error_response(f"Unknown youtuvl_mode: {mode}")
        except Exception as e:
            logger.exception(f"Error in YoutuVL {mode} mode")
            return self.create_error_response(str(e))

    async def _youtuvl_layout_detect(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Stage 1: Layout detection."""
        # Build layout detection request
        layout_request = self._build_youtuvl_layout_request(request)

        # Call model for generation
        response = await self._generate_youtuvl_response(layout_request, raw_request)
        if isinstance(response, ErrorResponse):
            return response

        # Parse output
        output_text = response.choices[0].message.content or ""
        elements = YoutuVLLayoutParser.parse_layout_output(output_text)

        # Filter by types
        layout_types = getattr(request, 'layout_types', None)
        if layout_types:
            elements = YoutuVLLayoutParser.filter_by_types(elements, layout_types)

        # Sort elements
        elements = YoutuVLLayoutParser.sort_elements(elements)

        # Build response
        return self._build_youtuvl_response(
            request_id=response.id,
            model=response.model,
            mode="layout",
            elements=elements,
            usage=response.usage
        )

    async def _youtuvl_ocr_recognize(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Stage 2: OCR recognition (requires regions parameter)."""
        regions = getattr(request, 'regions', None)
        if not regions:
            return self.create_error_response(
                "OCR mode requires 'regions' parameter. "
                "Example: [{'type': 'LAYOUT_TEXT', 'bbox': [x1, y1, x2, y2]}]"
            )

        # Build LayoutElement list from regions
        elements = []
        for r in regions:
            bbox = r.get("bbox", [])
            if isinstance(bbox, dict):
                bbox = [bbox.get("x1", 0), bbox.get("y1", 0),
                        bbox.get("x2", 0), bbox.get("y2", 0)]
            elements.append(LayoutElement(
                type=r.get("type", "LAYOUT_TEXT"),
                bbox=tuple(bbox)
            ))

        # Batch OCR
        # Sean 的 SDK 默认 batch_size=1（更稳定，避免 <sep> 丢失导致结果错位）。
        batch_size = getattr(request, 'ocr_batch_size', 1) or 1
        elements = await self._batch_ocr(request, elements, batch_size, raw_request)

        return self._build_youtuvl_response(
            request_id=f"chatcmpl-{random_uuid()}",
            model=request.model or "youtuvl",
            mode="ocr",
            elements=elements,
            usage=None
        )

    async def _youtuvl_document_parse(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Full document parsing: Layout + OCR two-stage."""
        import time
        debug = self._YOUTUVL_DEBUG
        timing_stats = {} if debug else None
        total_start = time.perf_counter() if debug else 0

        # Check parsing mode
        parse_mode = getattr(request, 'parse_mode', 'sequential')  # 'sequential', 'streaming', 'single_pass'
        
        # Legacy support for streaming_layout
        streaming_layout = getattr(request, 'streaming_layout', False)
        if streaming_layout and parse_mode == 'sequential':
            parse_mode = 'streaming'
        
        if parse_mode == 'single_pass':
            # 最优化模式：Layout + 全部 OCR 在一次多轮对话中完成
            return await self._youtuvl_document_parse_single_pass(
                request, raw_request, timing_stats, total_start)
        elif parse_mode == 'streaming':
            return await self._youtuvl_document_parse_streaming(
                request, raw_request, timing_stats, total_start)
        
        # Default: sequential
        return await self._youtuvl_document_parse_sequential(
            request, raw_request, timing_stats, total_start)

    async def _youtuvl_document_parse_single_pass(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request],
        timing_stats: Optional[dict],
        total_start: float,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Single-pass optimization: Layout + All OCR with shared image encoding.
        
        Key insight: Use multi-turn conversation to keep image in context.
        - Turn 1: Layout detection
        - Turn 2: Batch OCR (all elements in one request)
        
        This avoids re-encoding the image for each OCR request.
        Expected speedup: 2-3x for documents with many elements.
        """
        import time
        debug = timing_stats is not None
        
        if debug:
            timing_stats["mode"] = "single_pass"
            stage1_start = time.perf_counter()

        # Stage 1: Layout detection
        layout_request = self._build_youtuvl_layout_request(request)
        
        if debug:
            layout_gen_start = time.perf_counter()
            
        layout_response = await self._generate_youtuvl_response(
            layout_request, raw_request)
        
        if debug:
            layout_gen_time = (time.perf_counter() - layout_gen_start) * 1000
            timing_stats["layout_generate_ms"] = round(layout_gen_time, 2)
            
        if isinstance(layout_response, ErrorResponse):
            return layout_response
            
        layout_output = layout_response.choices[0].message.content or ""
        elements = YoutuVLLayoutParser.parse_layout_output(layout_output)
        
        # Filter and sort
        layout_types = getattr(request, 'layout_types', None)
        if layout_types:
            elements = YoutuVLLayoutParser.filter_by_types(elements, layout_types)
        elements = YoutuVLLayoutParser.sort_elements(elements)
        
        if debug:
            timing_stats["layout_elements_count"] = len(elements)
            
        if not elements:
            if debug:
                total_time = (time.perf_counter() - total_start) * 1000
                timing_stats["total_ms"] = round(total_time, 2)
                logger.info(f"[YoutuVL DEBUG] Document parse timing (single_pass, no elements): {timing_stats}")
            return self._build_youtuvl_response(
                request_id=layout_response.id,
                model=layout_response.model,
                mode="document",
                elements=[],
                usage=layout_response.usage
            )
        
        # Separate elements: OCR vs skip
        skip_ocr_types = getattr(request, 'skip_ocr_types', None)
        if skip_ocr_types is None:
            SKIP_OCR_TYPES = {"LAYOUT_FIGURE", "LAYOUT_CHART", "LAYOUT_SEAL"}
        else:
            SKIP_OCR_TYPES = set(skip_ocr_types)
            
        ocr_elements = []
        for elem in elements:
            if elem.type in SKIP_OCR_TYPES:
                elem.text = f"[{elem.type.replace('LAYOUT_', '')}]"
            else:
                ocr_elements.append(elem)
        
        if not ocr_elements:
            if debug:
                total_time = (time.perf_counter() - total_start) * 1000
                timing_stats["total_ms"] = round(total_time, 2)
                logger.info(f"[YoutuVL DEBUG] Document parse timing (single_pass, no OCR needed): {timing_stats}")
            return self._build_youtuvl_response(
                request_id=layout_response.id,
                model=layout_response.model,
                mode="document",
                elements=elements,
                usage=layout_response.usage
            )
        
        # Stage 2: Single OCR request with ALL elements using multi-turn
        # This keeps the image in KV cache!
        if debug:
            ocr_start = time.perf_counter()
            timing_stats["ocr_elements_count"] = len(ocr_elements)
        
        # Build multi-turn conversation:
        # User: <image> + layout_prompt
        # Assistant: layout_output
        # User: ocr_prompt (all elements)
        ocr_prompt = YoutuVLLayoutParser.format_ocr_prompt(ocr_elements)
        
        # Get original image content
        original_user_content = []
        for msg in request.messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", [])
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", [])
            if role == "user":
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            original_user_content.append(item)
                        elif hasattr(item, "type") and item.type == "image_url":
                            original_user_content.append(item.model_dump() if hasattr(item, "model_dump") else item)
                break
        
        # Build multi-turn messages
        multi_turn_messages = [
            {
                "role": "user",
                "content": original_user_content + [{"type": "text", "text": YoutuVLLayoutParser.LAYOUT_PROMPT}]
            },
            {
                "role": "assistant", 
                "content": layout_output
            },
            {
                "role": "user",
                "content": ocr_prompt
            }
        ]
        
        ocr_request = ChatCompletionRequest(
            model=request.model,
            messages=multi_turn_messages,
            max_tokens=min(getattr(request, 'max_tokens', None) or 4096, 4096),
            temperature=0,
            top_p=getattr(request, 'top_p', None) or 0.3,
            repetition_penalty=getattr(request, 'repetition_penalty', None),
            stop=getattr(request, 'stop', None),
            stop_token_ids=getattr(request, 'stop_token_ids', None),
            stream=False,
            mm_processor_kwargs=getattr(request, 'mm_processor_kwargs', None),
        )
        
        ocr_response = await self._generate_youtuvl_response(ocr_request, raw_request)
        
        if debug:
            ocr_gen_time = (time.perf_counter() - ocr_start) * 1000
            timing_stats["ocr_generate_ms"] = round(ocr_gen_time, 2)
            if ocr_response and hasattr(ocr_response, 'usage') and ocr_response.usage:
                timing_stats["ocr_prompt_tokens"] = ocr_response.usage.prompt_tokens or 0
                timing_stats["ocr_output_tokens"] = ocr_response.usage.completion_tokens or 0
                if ocr_response.usage.prompt_tokens_details:
                    cached = ocr_response.usage.prompt_tokens_details.cached_tokens
                    if cached:
                        timing_stats["ocr_cached_tokens"] = cached
        
        if isinstance(ocr_response, ErrorResponse):
            # Fallback: fill empty text
            for elem in ocr_elements:
                elem.text = ""
        else:
            ocr_output = ocr_response.choices[0].message.content or ""
            texts = YoutuVLLayoutParser.parse_ocr_output(ocr_output, len(ocr_elements))
            for elem, text in zip(ocr_elements, texts):
                elem.text = text
        
        if debug:
            total_time = (time.perf_counter() - total_start) * 1000
            timing_stats["total_ms"] = round(total_time, 2)
            logger.info(f"[YoutuVL DEBUG] Document parse timing (single_pass): {timing_stats}")
        
        return self._build_youtuvl_response(
            request_id=layout_response.id,
            model=layout_response.model,
            mode="document",
            elements=elements,
            usage=layout_response.usage
        )

    async def _youtuvl_document_parse_sequential(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request],
        timing_stats: Optional[dict],
        total_start: float,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Optimized sequential implementation: Layout + OCR in single/few requests.
        
        Key optimization: Use large OCR batch to minimize image re-encoding.
        """
        import time
        debug = timing_stats is not None

        # Stage 1: Layout detection
        if debug:
            stage1_start = time.perf_counter()

        layout_request = self._build_youtuvl_layout_request(request)

        if debug:
            layout_build_time = (time.perf_counter() - stage1_start) * 1000
            timing_stats["layout_request_build_ms"] = round(layout_build_time, 2)
            layout_gen_start = time.perf_counter()

        layout_response = await self._generate_youtuvl_response(
            layout_request, raw_request)

        if debug:
            layout_gen_time = (time.perf_counter() - layout_gen_start) * 1000
            timing_stats["layout_generate_ms"] = round(layout_gen_time, 2)

        if isinstance(layout_response, ErrorResponse):
            return layout_response

        if debug:
            parse_start = time.perf_counter()
            # Get layout token stats
            if layout_response.usage:
                timing_stats["layout_prompt_tokens"] = layout_response.usage.prompt_tokens or 0
                timing_stats["layout_output_tokens"] = layout_response.usage.completion_tokens or 0
                # Get cached tokens if available (requires --enable-prompt-tokens-details)
                if layout_response.usage.prompt_tokens_details:
                    cached = layout_response.usage.prompt_tokens_details.cached_tokens
                    if cached:
                        timing_stats["layout_cached_tokens"] = cached

        output_text = layout_response.choices[0].message.content or ""
        elements = YoutuVLLayoutParser.parse_layout_output(output_text)

        # Filter by types
        layout_types = getattr(request, 'layout_types', None)
        if layout_types:
            elements = YoutuVLLayoutParser.filter_by_types(elements, layout_types)

        # Sort elements
        elements = YoutuVLLayoutParser.sort_elements(elements)

        if debug:
            parse_time = (time.perf_counter() - parse_start) * 1000
            timing_stats["layout_parse_ms"] = round(parse_time, 2)
            timing_stats["layout_elements_count"] = len(elements)
            stage1_total = (time.perf_counter() - stage1_start) * 1000
            timing_stats["stage1_total_ms"] = round(stage1_total, 2)

        if not elements:
            if debug:
                total_time = (time.perf_counter() - total_start) * 1000
                timing_stats["total_ms"] = round(total_time, 2)
                logger.info(f"[YoutuVL DEBUG] Document parse timing (no elements): {timing_stats}")
            return self._build_youtuvl_response(
                request_id=layout_response.id,
                model=layout_response.model,
                mode="document",
                elements=[],
                usage=layout_response.usage
            )

        # Stage 2: Batch OCR
        if debug:
            stage2_start = time.perf_counter()

        # 优化：使用更大的 batch_size 减少请求数，从而减少图像重编码次数
        # 默认 batch_size=1 太保守，建议使用更大的值
        batch_size = getattr(request, 'ocr_batch_size', 1) or 1

        if debug:
            timing_stats["ocr_batch_size"] = batch_size
            timing_stats["ocr_num_batches"] = (len(elements) + batch_size - 1) // batch_size

        elements = await self._batch_ocr(request, elements, batch_size, raw_request, timing_stats)

        if debug:
            stage2_total = (time.perf_counter() - stage2_start) * 1000
            timing_stats["stage2_total_ms"] = round(stage2_total, 2)
            total_time = (time.perf_counter() - total_start) * 1000
            timing_stats["total_ms"] = round(total_time, 2)
            logger.info(f"[YoutuVL DEBUG] Document parse timing (sequential): {timing_stats}")

        return self._build_youtuvl_response(
            request_id=layout_response.id,
            model=layout_response.model,
            mode="document",
            elements=elements,
            usage=layout_response.usage
        )

    async def _youtuvl_document_parse_streaming(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request],
        timing_stats: Optional[dict],
        total_start: float,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Streaming implementation: OCR starts as soon as layout elements are parsed.
        
        This overlaps Layout generation with OCR requests, reducing total latency.
        """
        import time
        debug = timing_stats is not None
        
        if debug:
            stage1_start = time.perf_counter()
            timing_stats["mode"] = "streaming"

        # Build layout request with streaming enabled
        layout_request = self._build_youtuvl_layout_request(request)
        layout_request.stream = True
        
        # Clear youtuvl_mode to avoid recursion
        if hasattr(layout_request, 'youtuvl_mode'):
            object.__setattr__(layout_request, 'youtuvl_mode', None)

        # Get skip_ocr_types config
        skip_ocr_types = getattr(request, 'skip_ocr_types', None)
        if skip_ocr_types is None:
            SKIP_OCR_TYPES = {"LAYOUT_FIGURE", "LAYOUT_CHART", "LAYOUT_SEAL"}
        else:
            SKIP_OCR_TYPES = set(skip_ocr_types)

        batch_size = getattr(request, 'ocr_batch_size', 1) or 1
        layout_types = getattr(request, 'layout_types', None)
        ocr_concurrency = getattr(request, 'ocr_concurrency', None)  # Limit concurrent OCR tasks
        
        # Accumulators
        full_text = ""
        parsed_elements: list = []
        last_parsed_count = 0
        
        # OCR tasks management
        ocr_tasks: list = []
        ocr_element_indices: list = []  # Maps task index to element indices
        pending_ocr_elements: list = []  # Elements waiting to form a batch
        pending_ocr_indices: list = []   # Indices of pending elements
        
        # Semaphore for OCR concurrency control
        ocr_semaphore = asyncio.Semaphore(ocr_concurrency) if ocr_concurrency else None
        
        request_id = f"chatcmpl-{random_uuid()}"
        model_name = request.model or "youtuvl"
        
        if debug:
            layout_gen_start = time.perf_counter()
            ocr_submit_times = []

        async def submit_ocr_batch(elements_batch: list, indices: list):
            """Submit a batch of elements for OCR."""
            if not elements_batch:
                return
            
            if debug:
                ocr_submit_times.append(time.perf_counter() - layout_gen_start)
            
            ocr_prompt = YoutuVLLayoutParser.format_ocr_prompt(elements_batch)
            ocr_request = self._build_youtuvl_ocr_request(request, ocr_prompt)
            
            async def run_ocr():
                if ocr_semaphore:
                    async with ocr_semaphore:
                        return await self._generate_youtuvl_response(ocr_request, raw_request)
                else:
                    return await self._generate_youtuvl_response(ocr_request, raw_request)
            
            task = asyncio.create_task(run_ocr())
            ocr_tasks.append(task)
            ocr_element_indices.append((indices, elements_batch))

        try:
            # Stream layout generation
            result = await self.create_chat_completion(layout_request, raw_request)
            
            if isinstance(result, ErrorResponse):
                return result
            
            # Process streaming response
            async for chunk_str in result:
                if not chunk_str.startswith("data: "):
                    continue
                if chunk_str.strip() == "data: [DONE]":
                    break
                    
                try:
                    chunk_data = json.loads(chunk_str[6:])
                    delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        
                        # Try to parse new elements incrementally
                        new_elements = YoutuVLLayoutParser.parse_layout_output(full_text)
                        
                        # Check if we have new complete elements
                        if len(new_elements) > last_parsed_count:
                            for i in range(last_parsed_count, len(new_elements)):
                                elem = new_elements[i]
                                
                                # Apply type filter
                                if layout_types and elem.type not in layout_types:
                                    continue
                                
                                parsed_elements.append(elem)
                                elem_idx = len(parsed_elements) - 1
                                
                                # Check if this element needs OCR
                                if elem.type in SKIP_OCR_TYPES:
                                    elem.text = f"[{elem.type.replace('LAYOUT_', '')}]"
                                else:
                                    pending_ocr_elements.append(elem)
                                    pending_ocr_indices.append(elem_idx)
                                    
                                    # Submit batch when full
                                    if len(pending_ocr_elements) >= batch_size:
                                        await submit_ocr_batch(
                                            pending_ocr_elements[:], 
                                            pending_ocr_indices[:]
                                        )
                                        pending_ocr_elements.clear()
                                        pending_ocr_indices.clear()
                            
                            last_parsed_count = len(new_elements)
                            
                except json.JSONDecodeError:
                    continue

            # Submit any remaining elements
            if pending_ocr_elements:
                await submit_ocr_batch(pending_ocr_elements, pending_ocr_indices)

            if debug:
                layout_gen_time = (time.perf_counter() - layout_gen_start) * 1000
                timing_stats["layout_generate_ms"] = round(layout_gen_time, 2)
                timing_stats["layout_elements_count"] = len(parsed_elements)
                timing_stats["ocr_tasks_submitted"] = len(ocr_tasks)
                if ocr_submit_times:
                    timing_stats["first_ocr_submit_ms"] = round(ocr_submit_times[0] * 1000, 2)

            # Sort elements by position
            parsed_elements = YoutuVLLayoutParser.sort_elements(parsed_elements)

            if not parsed_elements:
                if debug:
                    total_time = (time.perf_counter() - total_start) * 1000
                    timing_stats["total_ms"] = round(total_time, 2)
                    logger.info(f"[YoutuVL DEBUG] Document parse timing (streaming, no elements): {timing_stats}")
                return self._build_youtuvl_response(
                    request_id=request_id,
                    model=model_name,
                    mode="document",
                    elements=[],
                    usage=None
                )

            # Wait for all OCR tasks to complete
            if debug:
                ocr_wait_start = time.perf_counter()
                ocr_complete_times = []
                    
            if ocr_tasks:
                # Process results as they complete for better debugging
                completed_count = 0
                for coro in asyncio.as_completed(ocr_tasks):
                    try:
                        result = await coro
                        if debug:
                            ocr_complete_times.append((time.perf_counter() - ocr_wait_start) * 1000)
                    except Exception as e:
                        logger.warning(f"OCR task failed: {e}")
                        if debug:
                            ocr_complete_times.append((time.perf_counter() - ocr_wait_start) * 1000)
                    completed_count += 1
                
                # Now process all results (tasks are already done)
                for task_idx, task in enumerate(ocr_tasks):
                    indices, batch_elements = ocr_element_indices[task_idx]
                    
                    try:
                        result = task.result()
                    except Exception as e:
                        logger.warning(f"OCR task {task_idx} failed: {e}")
                        for elem in batch_elements:
                            elem.text = ""
                        continue
                    
                    if isinstance(result, ErrorResponse):
                        for elem in batch_elements:
                            elem.text = ""
                        continue
                    
                    output_text = result.choices[0].message.content or ""
                    texts = YoutuVLLayoutParser.parse_ocr_output(output_text, len(batch_elements))
                    
                    for elem, text in zip(batch_elements, texts):
                        elem.text = text

            if debug:
                ocr_wait_time = (time.perf_counter() - ocr_wait_start) * 1000
                timing_stats["ocr_wait_ms"] = round(ocr_wait_time, 2)
                total_time = (time.perf_counter() - total_start) * 1000
                timing_stats["total_ms"] = round(total_time, 2)
                # Calculate overlap benefit
                if ocr_submit_times:
                    overlap_time = layout_gen_time - (ocr_submit_times[0] * 1000)
                    timing_stats["layout_ocr_overlap_ms"] = round(overlap_time, 2)
                # Show OCR completion distribution
                if ocr_complete_times:
                    ocr_complete_times.sort()
                    timing_stats["ocr_first_complete_ms"] = round(ocr_complete_times[0], 2)
                    timing_stats["ocr_last_complete_ms"] = round(ocr_complete_times[-1], 2)
                    if len(ocr_complete_times) > 1:
                        timing_stats["ocr_median_complete_ms"] = round(ocr_complete_times[len(ocr_complete_times)//2], 2)
                timing_stats["ocr_concurrency"] = ocr_concurrency or "unlimited"
                logger.info(f"[YoutuVL DEBUG] Document parse timing (streaming): {timing_stats}")

            return self._build_youtuvl_response(
                request_id=request_id,
                model=model_name,
                mode="document",
                elements=parsed_elements,
                usage=None
            )
            
        except Exception as e:
            logger.exception(f"Error in streaming layout parse: {e}")
            # Fallback to sequential mode
            if debug:
                timing_stats["streaming_fallback"] = True
            return await self._youtuvl_document_parse_sequential(
                request, raw_request, timing_stats, total_start)

    async def _batch_ocr(
        self,
        request: ChatCompletionRequest,
        elements: list,
        batch_size: int,
        raw_request: Optional[Request] = None,
        timing_stats: Optional[dict] = None,
    ) -> list:
        """Batch OCR recognition with controlled parallel execution.
        
        OCR batches are submitted with controlled concurrency.
        Use ocr_concurrency to balance between throughput and latency.
        """
        import time
        debug = timing_stats is not None
        
        # Get concurrency limit from request
        ocr_concurrency = getattr(request, 'ocr_concurrency', None)
        
        # Types that don't need OCR (just description)
        # Can be overridden by request.skip_ocr_types
        skip_ocr_types = getattr(request, 'skip_ocr_types', None)
        if skip_ocr_types is None:
            # Default: skip FIGURE, CHART, SEAL
            SKIP_OCR_TYPES = {"LAYOUT_FIGURE", "LAYOUT_CHART", "LAYOUT_SEAL"}
        else:
            SKIP_OCR_TYPES = set(skip_ocr_types)
        
        # Separate elements: ones that need OCR vs ones that don't
        ocr_elements = []
        skip_elements = []
        for elem in elements:
            if elem.type in SKIP_OCR_TYPES:
                elem.text = f"[{elem.type.replace('LAYOUT_', '')}]"
                skip_elements.append(elem)
            else:
                ocr_elements.append(elem)
        
        if debug and skip_elements:
            timing_stats["ocr_skipped_count"] = len(skip_elements)
        
        # If no elements need OCR, return early
        if not ocr_elements:
            return elements
        
        # Prepare all batches (only for elements that need OCR)
        batches = []
        for i in range(0, len(ocr_elements), batch_size):
            batch = ocr_elements[i:i + batch_size]
            batch_idx = i // batch_size
            ocr_prompt = YoutuVLLayoutParser.format_ocr_prompt(batch)
            ocr_request = self._build_youtuvl_ocr_request(request, ocr_prompt)
            batches.append((batch_idx, batch, ocr_request))
        
        batch_results = []
        
        async def process_single_batch(batch_idx: int, batch: list, ocr_request):
            """Process a single OCR batch."""
            batch_start = time.perf_counter() if debug else 0
            
            # Call model
            response = await self._generate_youtuvl_response(
                ocr_request, raw_request)
            
            gen_time = (time.perf_counter() - batch_start) * 1000 if debug else 0
            
            if isinstance(response, ErrorResponse):
                # OCR failed, fill with empty text
                for elem in batch:
                    elem.text = ""
                if debug:
                    return {
                        "batch": batch_idx,
                        "elements": len(batch),
                        "generate_ms": round(gen_time, 2),
                        "error": True
                    }
                return None
            
            output_text = response.choices[0].message.content or ""
            
            # Parse output
            texts = YoutuVLLayoutParser.parse_ocr_output(output_text, len(batch))
            
            # Fill text
            for elem, text in zip(batch, texts):
                elem.text = text
            
            if debug:
                total_batch_time = (time.perf_counter() - batch_start) * 1000
                output_tokens = 0
                cached_tokens = 0
                prompt_tokens = 0
                if response.usage:
                    output_tokens = response.usage.completion_tokens or 0
                    prompt_tokens = response.usage.prompt_tokens or 0
                    if response.usage.prompt_tokens_details:
                        cached_tokens = response.usage.prompt_tokens_details.cached_tokens or 0
                batch_info = {
                    "batch": batch_idx,
                    "elements": len(batch),
                    "generate_ms": round(gen_time, 2),
                    "total_ms": round(total_batch_time, 2),
                    "output_tokens": output_tokens,
                    "output_chars": len(output_text),
                }
                if cached_tokens > 0:
                    batch_info["cached_tokens"] = cached_tokens
                    batch_info["prompt_tokens"] = prompt_tokens
                # Debug: show raw output and parsed texts
                if len(batch) > 1:
                    sep_count = output_text.count("<sep>")
                    batch_info["sep_count"] = sep_count
                    batch_info["expected_seps"] = len(batch) - 1
                    # Show first 200 chars of raw output for debugging
                    batch_info["raw_output_preview"] = output_text[:200] if len(output_text) > 200 else output_text
                return batch_info
            return None
        
        if ocr_concurrency is None or ocr_concurrency >= len(batches):
            # Full parallel execution
            tasks = [process_single_batch(idx, batch, req) for idx, batch, req in batches]
            batch_results = await asyncio.gather(*tasks)
        else:
            # Controlled concurrency using semaphore
            semaphore = asyncio.Semaphore(ocr_concurrency)
            
            async def limited_process(idx, batch, req):
                async with semaphore:
                    return await process_single_batch(idx, batch, req)
            
            tasks = [limited_process(idx, batch, req) for idx, batch, req in batches]
            batch_results = await asyncio.gather(*tasks)
        
        if debug:
            batch_times = [r for r in batch_results if r is not None]
            # Sort by batch index for consistent output
            batch_times.sort(key=lambda x: x["batch"])
            
            if batch_times:
                timing_stats["ocr_batches"] = batch_times
                timing_stats["ocr_concurrency"] = ocr_concurrency or len(batches)
                # Summary stats
                gen_times = [b["generate_ms"] for b in batch_times if "generate_ms" in b]
                output_tokens_list = [b.get("output_tokens", 0) for b in batch_times]
                cached_tokens_list = [b.get("cached_tokens", 0) for b in batch_times]
                if gen_times:
                    timing_stats["ocr_generate_total_ms"] = round(sum(gen_times), 2)
                    timing_stats["ocr_generate_avg_ms"] = round(sum(gen_times) / len(gen_times), 2)
                    timing_stats["ocr_output_tokens_total"] = sum(output_tokens_list)
                    if sum(output_tokens_list) > 0:
                        timing_stats["ocr_ms_per_token"] = round(sum(gen_times) / sum(output_tokens_list), 2)
                    # Add cached tokens summary if available
                    if any(cached_tokens_list):
                        timing_stats["ocr_cached_tokens_total"] = sum(cached_tokens_list)
                        timing_stats["ocr_cached_tokens_avg"] = round(sum(cached_tokens_list) / len(cached_tokens_list), 2)

        return elements

    def _build_youtuvl_layout_request(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        """Build layout detection request."""
        new_messages = []
        for msg in request.messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", [])
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", [])

            if role == "user":
                new_content = []
                # Keep image content
                if isinstance(content, list):
                    for item in content:
                        # Handle both dict and pydantic model
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                new_content.append(item)
                        elif hasattr(item, "type"):
                            # Pydantic model
                            if item.type == "image_url":
                                new_content.append(item.model_dump() if hasattr(item, "model_dump") else item)
                elif isinstance(content, str):
                    # content is just a string, no image
                    logger.info(f"[YoutuVL] _build_youtuvl_layout_request: content is string, no image")

                # Add layout prompt
                new_content.append({
                    "type": "text",
                    "text": YoutuVLLayoutParser.LAYOUT_PROMPT
                })
                new_messages.append({"role": "user", "content": new_content})
            else:
                new_messages.append(
                    msg if isinstance(msg, dict) else msg.model_dump())

        # Create new request (reuse most parameters from original)
        # NOTE: keep stop/penalties from the original request. Losing these can
        # cause OCR/document decoding to "run away" and break <sep>-based parsing.
        # Layout 输出通常只有 ~100-200 tokens，限制 max_tokens 减少不必要的计算
        return ChatCompletionRequest(
            model=request.model,
            messages=new_messages,
            max_tokens=512,  # Layout 输出通常 < 200 tokens
            max_completion_tokens=getattr(request, 'max_completion_tokens', None),
            temperature=0,  # Use greedy for layout detection
            top_p=getattr(request, 'top_p', None) or 0.3,
            repetition_penalty=getattr(request, 'repetition_penalty', None),
            stop=getattr(request, 'stop', None),
            stop_token_ids=getattr(request, 'stop_token_ids', None),
            stream=False,
            mm_processor_kwargs=getattr(request, 'mm_processor_kwargs', None),
        )

    def _build_youtuvl_ocr_request(
        self,
        request: ChatCompletionRequest,
        ocr_prompt: str
    ) -> ChatCompletionRequest:
        """Build OCR request.
        
        Note: We do NOT include Layout prompt as prefix here because:
        1. The model would continue generating Layout output instead of OCR
        2. Prefix cache for multimodal models requires exact token match including image
        3. Each OCR request has different bbox coordinates anyway
        """
        new_messages = []
        for msg in request.messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", [])
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", [])

            if role == "user":
                new_content = []
                # Keep image content
                if isinstance(content, list):
                    for item in content:
                        # Handle both dict and pydantic model
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                new_content.append(item)
                        elif hasattr(item, "type"):
                            if item.type == "image_url":
                                new_content.append(item.model_dump() if hasattr(item, "model_dump") else item)

                # Only OCR prompt, no Layout prefix
                new_content.append({"type": "text", "text": ocr_prompt})
                new_messages.append({"role": "user", "content": new_content})
            else:
                new_messages.append(
                    msg if isinstance(msg, dict) else msg.model_dump())

        # NOTE: keep stop/penalties from the original request.
        # OCR 输出通常较短，限制 max_tokens 可以减少不必要的计算
        # 单个 OCR 区域通常 < 500 tokens，batch_size=2 时 < 1000 tokens
        ocr_max_tokens = min(getattr(request, 'max_tokens', None) or 4096, 2048)

        return ChatCompletionRequest(
            model=request.model,
            messages=new_messages,
            max_tokens=ocr_max_tokens,
            max_completion_tokens=getattr(request, 'max_completion_tokens', None),
            temperature=0,
            top_p=getattr(request, 'top_p', None) or 0.3,
            repetition_penalty=getattr(request, 'repetition_penalty', None),
            stop=getattr(request, 'stop', None),
            stop_token_ids=getattr(request, 'stop_token_ids', None),
            stream=False,
            mm_processor_kwargs=getattr(request, 'mm_processor_kwargs', None),
        )

    async def _generate_youtuvl_response(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Generate single response (non-streaming)."""
        # Ensure stream=False
        request.stream = False

        # Clear youtuvl_mode to avoid recursion
        if hasattr(request, 'youtuvl_mode'):
            object.__setattr__(request, 'youtuvl_mode', None)

        # Call original chat completion logic
        result = await self.create_chat_completion(request, raw_request)

        if isinstance(result, ErrorResponse):
            return result

        # If it's an AsyncGenerator, we need to collect the full response
        if hasattr(result, '__aiter__'):
            raise ValueError("Internal error: stream should be False for YoutuVL")

        return result

    def _build_youtuvl_response(
        self,
        request_id: str,
        model: str,
        mode: str,
        elements: list,
        usage: Optional[UsageInfo] = None
    ) -> ChatCompletionResponse:
        """Build YoutuVL response."""
        result = {
            "mode": mode,
            "elements": [
                {
                    "index": i,
                    **elem.to_dict()
                }
                for i, elem in enumerate(elements)
            ]
        }
        
        # Ensure usage is not None
        if usage is None:
            usage = UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0)

        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=json.dumps(result, ensure_ascii=False),
                    ),
                    finish_reason="stop"
                )
            ],
            usage=usage
        )

    # ========== End of YoutuVL Methods ==========