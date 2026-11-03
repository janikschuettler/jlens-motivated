"""Prompt token IDs for wave-2 arms: assistant prefill, multi-turn chats, and
chats whose earlier assistant turn is an on-policy sample (spliced as token IDs).

No heavy imports: callers pass a loaded HF tokenizer (gen env).

Qwen3's chat template (checked 2026-11-02 on the pinned revision):
  - thinking-mode generation prompt ends with "<|im_start|>assistant\\n"; the
    model itself emits "<think>" (ID 151667) as its first token;
  - non-thinking generation prompt ends with "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n";
  - an assistant turn in the history is rendered as
    "<|im_start|>assistant\\n" + content.split("</think>")[-1].lstrip("\\n") + "<|im_end|>\\n",
    i.e. earlier thinking is dropped and only the visible answer stays;
  - saved completions end with "<|im_end|>" (151645) in the token IDs, but vLLM
    leaves the stop string out of completion_text.

Everything here works on token IDs. Text is tokenized only for pieces we write
ourselves (prefill strings, follow-up user turns, template glue), and each
splice is checked against the tokenizer's own rendering of the same chat
(`template_equal`), so a BPE merge across a boundary is detected and recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jmr.tokens import chat_text, encode_chat, token_bytes

THINK_START, THINK_END, IM_END = "<think>", "</think>", "<|im_end|>"


@dataclass
class PromptPlan:
    """A prompt as token IDs plus how it was built. `messages` is the chat as a
    template would see it (for the record); `prompt_ids` is what is sent."""

    messages: list[dict]
    prompt_ids: list[int]
    enable_thinking: bool
    info: dict = field(default_factory=dict)


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def special_id(tokenizer, token: str) -> int:
    i = tokenizer.convert_tokens_to_ids(token)
    if i is None or i == tokenizer.unk_token_id:
        raise ValueError(f"{token!r} is not a single token")
    return i


def plain_prompt(tokenizer, messages: list[dict], *, enable_thinking: bool) -> PromptPlan:
    """The chat rendered by the template (single- or multi-turn, all text ours)."""
    return PromptPlan(messages, encode_chat(tokenizer, messages, enable_thinking=enable_thinking),
                      enable_thinking, {"kind": "template"})


def prefill_prompt(tokenizer, messages: list[dict], prefill: str) -> PromptPlan:
    """Thinking-mode prompt with the assistant turn prefilled (e.g.
    "<think>\\nThe answer is (B). Let me explain why."). The prefill belongs to
    the prompt IDs; the completion holds only generated tokens.

    Raises if tokenizing the prefill on its own differs from tokenizing the
    rendered prompt + prefill as one string (a merge across the boundary)."""
    head_text = chat_text(tokenizer, messages, enable_thinking=True)
    head = _ids(tokenizer, head_text)
    pre = _ids(tokenizer, prefill)
    joint = _ids(tokenizer, head_text + prefill)
    if head + pre != joint:
        raise ValueError(f"prefill {prefill!r} merges with the prompt when tokenized jointly")
    return PromptPlan(messages, head + pre, True, {
        "kind": "prefill", "prefill_text": prefill, "prefill_token_ids": pre, "n_prefill_tokens": len(pre),
        "prefill_starts_thinking": prefill.startswith(THINK_START),
    })


# ---- on-policy assistant turns ---------------------------------------------------


def visible_response_ids(tokenizer, completion_ids: list[int], *, thinking: bool) -> list[int]:
    """The generated token IDs of the visible answer: after the last "</think>"
    (thinking runs), without leading newline-only tokens (the template strips
    them) and without the final "<|im_end|>". Raises if there is no "</think>"
    in a thinking run or the run didn't end with "<|im_end|>"."""
    end_id, im_end = special_id(tokenizer, THINK_END), special_id(tokenizer, IM_END)
    ids = list(completion_ids)
    if not ids or ids[-1] != im_end:
        raise ValueError("completion does not end with <|im_end|> (truncated or other stop)")
    ids = ids[:-1]
    if thinking:
        if end_id not in ids:
            raise ValueError("thinking run without </think>")
        ids = ids[len(ids) - ids[::-1].index(end_id):]
    pieces = token_bytes(tokenizer, ids)
    k = 0
    while k < len(ids) and pieces[k] and set(pieces[k]) <= {ord("\n")}:
        k += 1
    if k < len(ids) and pieces[k].startswith(b"\n"):
        raise ValueError(f"response starts with a token mixing newlines and text: {pieces[k]!r}")
    return ids[k:]


def splice_assistant_turn(tokenizer, prefix_messages: list[dict], response_ids: list[int],
                          following: list[dict], *, enable_thinking: bool,
                          prefix_ids: list[int] | None = None) -> PromptPlan:
    """prefix chat + an assistant turn given as generated token IDs + more turns
    (e.g. a user follow-up), ending with the generation prompt.

    prompt IDs = prefix_ids (default: the template's IDs for prefix_messages with a
    thinking-mode generation prompt, i.e. ending "<|im_start|>assistant\\n")
    + response_ids + IDs of the template text after the assistant content (it
    starts with the special token "<|im_end|>", so it can't merge backwards).
    The template must render the decoded response unchanged (checked);
    `template_equal` says whether the result also equals the template's own
    tokenization of the whole chat (False means the sampled tokens are not the
    canonical BPE of their text; the spliced, on-policy IDs are used anyway)."""
    content = tokenizer.decode(response_ids)
    messages = list(prefix_messages) + [{"role": "assistant", "content": content}] + list(following)
    head_text = chat_text(tokenizer, prefix_messages, enable_thinking=True)
    full_text = chat_text(tokenizer, messages, enable_thinking=enable_thinking)
    if not full_text.startswith(head_text + content + IM_END):
        raise ValueError("the chat template does not render the assistant turn verbatim after the prefix")
    head = _ids(tokenizer, head_text) if prefix_ids is None else list(prefix_ids)
    tail = _ids(tokenizer, full_text[len(head_text) + len(content):])
    ids = head + list(response_ids) + tail
    template_ids = _ids(tokenizer, full_text)
    return PromptPlan(messages, ids, enable_thinking, {
        "kind": "spliced", "assistant_span": [len(head), len(head) + len(response_ids)],
        "template_equal": ids == template_ids, "head_from_record": prefix_ids is not None,
        "head_matches_template": head == _ids(tokenizer, head_text),
    })
