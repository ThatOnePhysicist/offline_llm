"""Model access for the worker: chat formatting and generation on top of model_handler."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

Message = tuple[str, str]  # (role, content)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
# Text a chat template may already start with; adding another BOS token would double it.
_BOS_PREFIXES = ("<｜begin▁of▁sentence｜>", "<s>", "<|begin_of_text|>", "<|startoftext|>")


@dataclass
class GenerationSettings:
    ctx_size: int = 8192
    max_tokens: int = 4096
    temperature: float = 0.6
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.0


@dataclass
class Generation:
    text: str           # everything the model produced
    answer: str         # the same with <think>...</think> reasoning removed
    prompt_tokens: int
    tokens: int
    stop_reason: str    # eos | max_tokens | context | callback
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


def strip_reasoning(text: str) -> str:
    """Removes <think>...</think> blocks; an unfinished block is dropped to the end."""
    text = _THINK_BLOCK.sub("", text)
    start = text.find("<think>")
    if start != -1:
        text = text[:start]
    return text.replace("</think>", "").strip()


def chatml(messages: Sequence[Message]) -> str:
    """Qwen-style prompt, the same format chat_script_r.py builds."""
    parts = [f"<|im_start|>{role}\n{content}<|im_end|>" for role, content in messages]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def plain(messages: Sequence[Message]) -> str:
    """Model-agnostic fallback: 'User:' / 'Assistant:' turns."""
    out = []
    for role, content in messages:
        if role == "system":
            out.append(content.strip() + "\n")
        else:
            out.append(f"{'User' if role == 'user' else 'Assistant'}: {content.strip()}\n")
    out.append("Assistant:")
    return "\n".join(out)


class LocalModel:
    """A GGUF model loaded through the Rust ModelManager.

    template: "auto" uses the chat template stored in the GGUF file (falls back to ChatML if
    llama.cpp does not know it); "chatml" or "plain" force a Python-side format; any other
    value is passed to llama.cpp as a template name (for example "deepseek3").
    """

    def __init__(self, path: str | Path, name: str | None = None,
                 settings: GenerationSettings | None = None, template: str = "auto",
                 log: Callable[[str], None] = print) -> None:
        import model_handler  # the compiled Rust extension (make build)

        self.path = Path(path).expanduser()
        self.name = name or self.path.stem
        self.settings = settings or GenerationSettings()
        self.template = template
        self.log = log
        self.handler = model_handler.ModelManager()
        log(self.handler.load_model(self.name, str(self.path)))
        try:
            trained = int(self.handler.n_ctx_train())
            if trained and self.settings.ctx_size > trained:
                log(f"context {self.settings.ctx_size} is above what {self.name} was trained on; using {trained}")
                self.settings.ctx_size = trained
        except Exception:  # older backend without n_ctx_train
            pass
        self.template_used = self._pick_template()
        log(f"chat format: {self.template_used}")

    def _pick_template(self) -> str:
        if self.template in ("chatml", "plain"):
            return self.template
        probe = [("system", "s"), ("user", "u")]
        try:
            if self.template == "auto":
                self.handler.apply_chat_template(probe, True)
                return "gguf"
            self.handler.apply_chat_template(probe, True, self.template)
            return self.template
        except Exception as err:
            self.log(f"chat template not usable ({err}); using chatml")
            return "chatml"

    def format(self, messages: Sequence[Message]) -> str:
        if self.template_used == "chatml":
            return chatml(messages)
        if self.template_used == "plain":
            return plain(messages)
        name = None if self.template_used == "gguf" else self.template_used
        return self.handler.apply_chat_template(list(messages), True, name)

    def chat(self, messages: Sequence[Message], should_stop: Callable[[], bool] = lambda: False,
             on_text: Callable[[str], None] | None = None) -> Generation:
        prompt = self.format(messages)
        add_bos = not prompt.startswith(_BOS_PREFIXES)
        pieces: list[str] = []

        def callback(piece: str):
            pieces.append(piece)
            if on_text is not None:
                on_text(piece)
            return False if should_stop() else None

        s = self.settings
        start = time.perf_counter()
        prompt_tokens, tokens, reason = self.handler.generate(
            prompt, callback,
            max_tokens=s.max_tokens, ctx_size=s.ctx_size, temperature=s.temperature,
            top_k=s.top_k, top_p=s.top_p, repeat_penalty=s.repeat_penalty, add_bos=add_bos,
        )
        text = "".join(pieces)
        return Generation(text, strip_reasoning(text), prompt_tokens, tokens, reason,
                          time.perf_counter() - start)

    def count_hint(self, text: str) -> int:
        """Rough token count used to budget prompts (Lean text runs about 3 characters per token)."""
        return len(text) // 3


@dataclass
class ScriptedModel:
    """Stand-in for tests and dry runs: returns canned answers in order, then repeats the last."""

    answers: list[str]
    name: str = "scripted"
    settings: GenerationSettings = field(default_factory=GenerationSettings)
    calls: list[list[Message]] = field(default_factory=list)

    def chat(self, messages: Sequence[Message], should_stop: Callable[[], bool] = lambda: False,
             on_text: Callable[[str], None] | None = None) -> Generation:
        self.calls.append(list(messages))
        text = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return Generation(text, strip_reasoning(text), 0, len(text) // 3, "eos", 0.0)

    def count_hint(self, text: str) -> int:
        return len(text) // 3

    @classmethod
    def from_file(cls, path: str | Path) -> "ScriptedModel":
        """Answers separated by lines consisting of '====='."""
        chunks = re.split(r"(?m)^=====\s*$", Path(path).read_text(encoding="utf-8"))
        return cls([c.strip("\n") for c in chunks if c.strip()])
