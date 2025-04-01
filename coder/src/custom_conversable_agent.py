from autogen import ConversableAgent
from autogen.agentchat import ChatResult
from typing import Any, Callable, Dict, Optional, Union
from autogen.cache.cache import AbstractCache
from autogen.agentchat.utils import consolidate_chat_info, gather_usage_summary

class CustomConversableAgent(ConversableAgent):

    DEFAULT_SUMMARY_METHOD = "last_msg"

    def initiate_chat_with_itr(
        self,
        recipient: "ConversableAgent",
        clear_history: bool = True,
        silent: Optional[bool] = False,
        cache: Optional[AbstractCache] = None,
        max_turns: Optional[int] = None,
        summary_method: Optional[Union[str, Callable]] = DEFAULT_SUMMARY_METHOD,
        summary_args: Optional[dict] = {},
        message: Optional[Union[Dict, str, Callable]] = None,
        **kwargs,
    ) -> ChatResult:
        """Initiate a chat with the recipient agent.

        Reset the consecutive auto reply counter.
        If `clear_history` is True, the chat history with the recipient agent will be cleared.


        Args:
            recipient: the recipient agent.
            clear_history (bool): whether to clear the chat history with the agent. Default is True.
            silent (bool or None): (Experimental) whether to print the messages for this conversation. Default is False.
            cache (AbstractCache or None): the cache client to be used for this conversation. Default is None.
            max_turns (int or None): the maximum number of turns for the chat between the two agents. One turn means one conversation round trip. Note that this is different from
                [max_consecutive_auto_reply](#max_consecutive_auto_reply) which is the maximum number of consecutive auto replies; and it is also different from [max_rounds in GroupChat](./groupchat#groupchat-objects) which is the maximum number of rounds in a group chat session.
                If max_turns is set to None, the chat will continue until a termination condition is met. Default is None.
            summary_method (str or callable): a method to get a summary from the chat. Default is DEFAULT_SUMMARY_METHOD, i.e., "last_msg".

            Supported strings are "last_msg" and "reflection_with_llm":
                - when set to "last_msg", it returns the last message of the dialog as the summary.
                - when set to "reflection_with_llm", it returns a summary extracted using an llm client.
                    `llm_config` must be set in either the recipient or sender.

            A callable summary_method should take the recipient and sender agent in a chat as input and return a string of summary. E.g.,

            ```python
            def my_summary_method(
                sender: ConversableAgent,
                recipient: ConversableAgent,
                summary_args: dict,
            ):
                return recipient.last_message(sender)["content"]
            ```
            summary_args (dict): a dictionary of arguments to be passed to the summary_method.
                One example key is "summary_prompt", and value is a string of text used to prompt a LLM-based agent (the sender or receiver agent) to reflect
                on the conversation and extract a summary when summary_method is "reflection_with_llm".
                The default summary_prompt is DEFAULT_SUMMARY_PROMPT, i.e., "Summarize takeaway from the conversation. Do not add any introductory phrases. If the intended request is NOT properly addressed, please point it out."
                Another available key is "summary_role", which is the role of the message sent to the agent in charge of summarizing. Default is "system".
            message (str, dict or Callable): the initial message to be sent to the recipient. Needs to be provided. Otherwise, input() will be called to get the initial message.
                - If a string or a dict is provided, it will be used as the initial message.        `generate_init_message` is called to generate the initial message for the agent based on this string and the context.
                    If dict, it may contain the following reserved fields (either content or tool_calls need to be provided).

                        1. "content": content of the message, can be None.
                        2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                        3. "tool_calls": a list of dictionaries containing the function name and arguments.
                        4. "role": role of the message, can be "assistant", "user", "function".
                            This field is only needed to distinguish between "function" or "assistant"/"user".
                        5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                        6. "context" (dict): the context of the message, which will be passed to
                            [OpenAIWrapper.create](../oai/client#create).

                - If a callable is provided, it will be called to get the initial message in the form of a string or a dict.
                    If the returned type is dict, it may contain the reserved fields mentioned above.

                    Example of a callable message (returning a string):

            ```python
            def my_message(sender: ConversableAgent, recipient: ConversableAgent, context: dict) -> Union[str, Dict]:
                carryover = context.get("carryover", "")
                if isinstance(message, list):
                    carryover = carryover[-1]
                final_msg = "Write a blogpost." + "\\nContext: \\n" + carryover
                return final_msg
            ```

                    Example of a callable message (returning a dict):

            ```python
            def my_message(sender: ConversableAgent, recipient: ConversableAgent, context: dict) -> Union[str, Dict]:
                final_msg = {}
                carryover = context.get("carryover", "")
                if isinstance(message, list):
                    carryover = carryover[-1]
                final_msg["content"] = "Write a blogpost." + "\\nContext: \\n" + carryover
                final_msg["context"] = {"prefix": "Today I feel"}
                return final_msg
            ```
            **kwargs: any additional information. It has the following reserved fields:
                - "carryover": a string or a list of string to specify the carryover information to be passed to this chat.
                    If provided, we will combine this carryover (by attaching a "context: " string and the carryover content after the message content) with the "message" content when generating the initial chat
                    message in `generate_init_message`.
                - "verbose": a boolean to specify whether to print the message and carryover in a chat. Default is False.

        Raises:
            RuntimeError: if any async reply functions are registered and not ignored in sync chat.

        Returns:
            ChatResult: an ChatResult object.
        """
        _chat_info = locals().copy()
        _chat_info["sender"] = self
        consolidate_chat_info(_chat_info, uniform_sender=self)
        for agent in [self, recipient]:
            agent._raise_exception_on_async_reply_functions()
            agent.previous_cache = agent.client_cache
            agent.client_cache = cache
        if isinstance(max_turns, int):
            self._prepare_chat(recipient, clear_history, reply_at_receive=False)
            for itr in range(max_turns):
                if itr == 0:
                    if isinstance(message, Callable):
                        msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
                    else:
                        msg2send = self.generate_init_message(message, **kwargs)
                else:
                    self.chat_messages[recipient][-1]['content']+=f'\nThe {itr}th reply'
                    msg2send = self.generate_reply(messages=self.chat_messages[recipient], sender=recipient)
                if msg2send is None:
                    break
                self.send(msg2send, recipient, request_reply=True, silent=silent)
        else:
            self._prepare_chat(recipient, clear_history)
            if isinstance(message, Callable):
                msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
            else:
                msg2send = self.generate_init_message(message, **kwargs)
            self.send(msg2send, recipient, silent=silent)
        summary = self._summarize_chat(
            summary_method,
            summary_args,
            recipient,
            cache=cache,
        )
        for agent in [self, recipient]:
            agent.client_cache = agent.previous_cache
            agent.previous_cache = None
        chat_result = ChatResult(
            chat_history=self.chat_messages[recipient],
            summary=summary,
            cost=gather_usage_summary([self, recipient]),
            human_input=self._human_input,
        )
        return chat_result