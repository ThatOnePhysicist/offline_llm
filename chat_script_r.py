import model_handler  # Rust lib.rs
from conversation_sync import (
    sync_on_save,
    manual_sync_all,
    list_server_conversations,
    download_from_server,
    check_server_status
)
import logging
import os
import sys
import time
import json
import uuid
from datetime import datetime
from pathlib import Path

try:
    import pyperclip
    _CLIPBOARD_AVAILABLE = True
except ImportError:
    _CLIPBOARD_AVAILABLE = False

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.style import Style
from rich.text import Text
from rich.table import Table

"""
Chat Script Frontend for Offline LLM Inference.

This module provides a Rich-based terminal interface that communicates with a 
Rust backend (model_handler) to run GGUF models locally using Metal acceleration.
It supports dynamic model swapping between 'Coding' and 'Reasoning' models
based on the user's input.
"""

def _load_config():
    config_path = Path(__file__).parent / "config.toml"
    defaults = {
        "model": {
            "default_name" : "Qwen2.5-Coder-7B",
            "default_path" : "",
            "path_7b" : "",
            "path_14b" : "",
        },
        "context" : {"max_tokens": 7400},
        "server" : {"url": "", "enabled": False, "auto_sync": False},
        "logging" : {"level": "WARNING", "log_file": "llm_chat.log"}
    }

    if not config_path.exists():
        print(
            f"[warn] config.toml not found at {config_path}. Using defaults. \n\
                    Create config.toml next to this script to configure paths.",
            file = sys.stderr,
        )
        return defaults

    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        
        with open(config_path, "rb") as f:
            loaded = tomllib.load(f)
    except Exception as exc:
        print(f"[warn] Failed to parse config.toml: {exc}. Using defaults.", file = sys.stderr)
        return defaults

    for section, values in loaded.items():
        if isinstance(values, dict) and isinstance(defaults.get(section), dict):
            defaults[section].update(values)
        else:
            defaults[section] = values
    
    return defaults

CONFIG = _load_config()

_log_level = getattr(logging, CONFIG["logging"].get("level", "WARNING").upper(), logging.WARNING)
_log_file = CONFIG["logging"].get("log_file", "llm_chat.log")

logging.basicConfig(
    filename = Path(__file__).parent / _log_file,
    level = _log_level,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("llm_chat")

###########
# Globals #
###########
REPLY_TOKENS = 1024  # must match generate()'s max_tokens
MSG_OVERHEAD = 4     # <|im_start|>role\n ... <|im_end|>\n
console = Console()

# Local path configs
base_path = os.path.dirname(os.path.abspath(__file__))

# Conversation storage path
CONVERSATION_DIR = Path(base_path) / "conversations"
CONVERSATION_DIR.mkdir(exist_ok = True)

# Global state for the model engine 
handler = None
_current_model_name = None

# Conversation state
MAX_CONTEXT_TOKENS = CONFIG["context"].get("max_tokens", 7400)

current_temperature: float = 0.7
last_assistant_response: str = ""

class ConversationManager:
    """Manages conversation history, context window, and persistence."""

    def __init__(self, max_tokens = MAX_CONTEXT_TOKENS) -> None:
        self.history : list = []
        self.system_prompt : str = ""
        self.max_tokens : int = max_tokens
        self.current_file : str |  None = None
        self.saved_message_ids : set = set() # tracks which messages are already saved.
        self._prompt_locked : bool = False

    def set_system_prompt(self, prompt: str, force: bool = False) -> None:
        """
        
        """
        if self._prompt_locked and not force:
            return
        self.system_prompt = prompt
        self._prompt_locked = True

    def count_tokens(self, text: str) -> int:
        if handler is not None:
            try: 
                return handler.count_tokens(text)
            except Exception as exc:
                log.debug("Rust count_tokens failed, using heuristic: %s", exc)
        return len(text) // 4
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (approx. 1 token ~ 4 chars in English).
        Considering this is just for offline use, this should be okay. 
        If necessary, will look into using a tokenizer. 
        """
        return self.count_tokens(text)

    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.
        """
        self.history.append(
            {
                "id" : str(uuid.uuid4()),
                "role" : role,
                "content" : content,
                "timestamp" : datetime.now().isoformat()
            }
        )

        if role == "user" and self.is_near_limit(threshold = 0.85):
            new_message = self.history.pop()
            filepath = self.auto_save_and_clear(keep_last_n = 2)

            console.print(f"\n[yellow]Context limit nearing 85%[/yellow]")
            console.print(f"\n[green]Auto-saved to:[/green] [cyan]{filepath}[/cyan]")
            console.print(f"\n[dim]Keeping last 2 exchanges for continuty[/dim]")

            self.history.append(new_message)
    
    def is_near_limit(self, threshold = 0.85) -> bool:
        """
        Checks if approaching context limit.

        Parameters
        ----------
        threshold : float
            Trigger limit [default] 90%.
        """
        current_tokens = self.count_tokens("".join(msg["content"] for msg in self.history))
        return current_tokens >= self.max_tokens * threshold

    def get_trimmed_history(self) -> list:
        """
        Returns conversation history trimmed to fit within context window.
        Keeps most recent messages and ensures sytem prompt is always included.
        """
        system_tokens = self.estimate_tokens(self.system_prompt)
        available_tokens = self.max_tokens - system_tokens - REPLY_TOKENS

        trimmed = []
        token_count = 0

        for msg in reversed(self.history):
            msg_tokens = self.estimate_tokens(msg["content"]) + MSG_OVERHEAD
            if token_count + msg_tokens > available_tokens:
                break
            trimmed.insert(0, msg)
            token_count += msg_tokens
        
        return trimmed

    def build_prompt(self) -> str:
        """
        Build the full prompt with system and conversation history.
        """
        trimmed_history = self.get_trimmed_history()

        prompt_parts = [f"<|im_start|>system\n{self.system_prompt}<|im_end|>"]

        for msg in trimmed_history:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def clear(self, clear_file_tracking = True):
        """
        Clear conversation history.

        Parameters
        ----------
        clear_file_tracking : bool
            If True, starts fresh session, if False keeps current_file
        """
        self.history = []
        self.saved_message_ids = set()
        self._prompt_locked = False
        self.system_prompt = ""
        if clear_file_tracking:
            self.current_file = None

    def save(self, filename: str | None = None, append_mode: bool | None = None) -> str:
        """
        Save conversation to JSON file.

        Parameters
        ----------
        filename : str
            File name to save to. If None, uses current_file or creates new timestamp file.
        append_mode : bool
            If True, appends new file instead of overwriting. 
        """
        # Determine file name
        if filename is None:
            if self.current_file:
                filename = self.current_file
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.json"
        
        filepath = CONVERSATION_DIR / filename

        # Autodetect append_mode if not specified
        if append_mode is None:
            append_mode = (filename == self.current_file) and filepath.exists()
        
        # Handle append mode
        if append_mode and filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                
                existing_history = existing_data.get("history", [])

                existing_ids = {m.get("id") for m in existing_history if m.get("id")}

                new_messages = []
                for msg in self.history:
                    msg_id = msg.get("id")
                    if msg_id and msg_id in (self.saved_message_ids | existing_ids):
                        continue
                    if not msg_id:
                        ts = msg.get("timestamp")
                        if any(m.get("timestamp") == ts for m in existing_history):
                            continue
                    
                    new_messages.append(msg)
                    if msg_id:
                        self.saved_message_ids.add(msg_id)
                
                merged_history = existing_history + new_messages
                created = existing_data.get("created", datetime.now().isoformat())

                # print(f"[DEBUG] Append mode: {len(existing_history)} existing + {len(new_messages)} new = {len(merged_history)} total")

            except (json.JSONDecodeError, FileNotFoundError) as exc:
                log.warning("Append-mode read failed (%s); falling back to full save.", exc)
                merged_history = self.history
                created = datetime.now().isoformat()
                # print(f"[DEBUG] File corrupted/missing saving all {len(merged_history)} messages")
        else:
            merged_history = self.history
            created = datetime.now().isoformat()

            for msg in self.history:
                if msg_id := msg.get("id"):
                    self.saved_message_ids.add(msg_id)

            # print(f"[DEBUG] New save: {len(merged_history)} messages")
        
        data = {
            "created": created,
            "system_prompt": self.system_prompt,
            "history": merged_history,
            "max_tokens": self.max_tokens
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent = 2, ensure_ascii = False)
        
        self.current_file = filename

        return str(filepath)
    
    def auto_save_and_clear(self, keep_last_n = 2) -> str:
        """
        Autosaves conversation and keeps last N=2 exchanges for context.

        Parameters
        ----------
        keep_last_n : int
            Number of recent pairs to keep.
        
        Returns
        -------
        str
            Path to saved conversation
        """
        if self.current_file:
            filename = self.current_file
            append_mode = True
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"auto_save_{timestamp}.json"
            append_mode = False
        
        # print(f"[DEBUG] Auto-save before: {len(self.history)} messages in memory")

        filepath = self.save(filename, append_mode = append_mode)

        try:
            sync_on_save(filepath)
        except Exception as exc:
            log.warning(f"Server sync after auto-save failed: %s", exc)


        if keep_last_n > 0:
            keep_count = keep_last_n * 2
            kept_messages = self.history[-keep_count:] if len(self.history) > keep_count else self.history
            self.history = kept_messages
            self.saved_message_ids = set()
            # print(f"[DEBUG] Auto-save after: {len(self.history)} messages kept in memory")
        else:
            self.history = []
            self.saved_message_ids = set()
        
        return filepath

    def load(self, filename: str) -> bool:
        """
        Load history from JSON file.
        """
        filepath = CONVERSATION_DIR / filename

        if not filepath.exists():
            return False

        with open(filepath, 'r', encoding = "utf-8") as f:
            data = json.load(f)

        self.system_prompt = data.get("system_prompt", "")
        self.history = data.get("history", [])
        self.max_tokens = data.get("max_tokens", MAX_CONTEXT_TOKENS)
        self.current_file = filename
        self._prompt_locked = bool(self.system_prompt)
        self.saved_message_ids = {m["id"] for m in self.history if m.get("id")}

        return True

    def list_conversations(self) -> list:
        """
        List all saved conversations.
        """
        conversations = []

        for filepath in CONVERSATION_DIR.glob("*.json"):
            try:
                with open(filepath, "r", encoding = "utf-8") as f:
                    data = json.load(f)
                conversations.append(
                    {
                        "filename": filepath.name,
                        "created": data.get("created", "Unknown"),
                        "message_count": len(data.get("history", []))
                    }
                )
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Skipping unreadable conversation file %s: %s", filepath.name, exc)

        return sorted(conversations, key = lambda x: x["created"], reverse = True)

    def get_stats(self) -> dict:
        """
        Get conversation statistics.
        """
        total_messages = len(self.history)
        user_messages = sum(1 for msg in self.history if msg["role"] == "user")
        assistant_messages = sum(1 for msg in self.history if msg["role"] == "assistant")

        total_chars = sum(len(msg["content"]) for msg in self.history)
        estimated_tokens = self.count_tokens("".join(msg["content"] for msg in self.history))

        percentage_used = (estimated_tokens / self.max_tokens) * 100
        context_usage = f"{estimated_tokens}/{self.max_tokens}"

        if percentage_used >= 85:
            context_usage += "Nearing limit..."

        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "context_usage": context_usage,
            "percent_used": f"{percentage_used:.1f}%",
            "session_file": self.current_file if self.current_file else "[dim]None[/dim]",
        } 

# Global conversation manager
conv_manager = ConversationManager()

def load_model_once(model_path: str, model_name: str):
    """
    Uses the Rust ModelManager to swap models in VRAM.

    Parameters
    ----------
    model_id : str
        Path to GGUF file. 
    model_name : str
        Model name.
    """
    global _current_model_name, handler

    if _current_model_name == model_name:
        return True

    if handler is None:
        handler = model_handler.ModelManager()

    console.print(f"[bold yellow]Initializing engine for {model_name}...[/bold yellow]")

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File not found at: {model_path}")

        # Rust reads GGUF and maps tensors
        result = handler.load_model(model_name, model_path)
        console.print(f"[green]{result}[/green]")
        _current_model_name = model_name
        return True

    except Exception as exc:
        log.error(f"Failed to load model %s: %s", model_name, exc)
        console.print(f"[bold red]Load Error:[/bold red] {exc}")
        return False


def build_system_prompt(user_input):
    """
    Constructs the system prompt based on language based keywords or 'reason'.
    """
    lower = user_input.lower()

    if "rust" in lower:
        return "You are an experienced Rust developer. Write correct, idiomatic Rust code."
    if "kotlin" in lower:
        return "You are a Kotlin expert. Use idiomatic patterns and null safety."
    if "javascript" in lower or " js " in lower or "typescript" in lower:
        return "You are a JavaScript/TypeScript expert. Write modern, clean code."
    if "golang" in lower or " go " in lower:
        return "You are a Go expert. Write idiomatic Go code."
    if "java" in lower:
        return "You are a Java expert. Write clean, maintainable code."
    if "c++" in lower or "cpp" in lower:
        return "You are a C++ expert. Write efficient, modern code."
    if "bash" in lower or "shell" in lower or "zsh" in lower:
        return "You are a Unix shell expert. Write safe, POSIX-compliant scripts."
    
    return "You are an expert software engineer. Write clean, well-documented code."

def get_multiline_input() -> str:
    console.print("\n[dim]Type /help for commands | End line with \\ for multiline[/dim]")
    first_line = console.input("[bold green]User: [/bold green]")

    if not first_line.endswith("\\"):
        return first_line.strip()

    lines = [first_line.rstrip("\\")]
    console.print("[dim]Multiline mode - press Enter on an empty line to send[/dim]")

    while True:
        line = console.input("[dim]... [/dim]")
        if line == "":
            break
        lines.append(line)
    
    return "\n".join(lines).strip()

def handle_command(command: str) -> bool:
    """
    
    """
    
    global current_temperature, last_assistant_response

    parts = command.strip().split(maxsplit = 1)
    base = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None
    
    path_7b = CONFIG["model"].get("path_7b", "")
    path_14b = CONFIG["model"].get("path_14b", "")


    match base:

        case "/clear" | "/new":
            conv_manager.clear(clear_file_tracking = True)
            console.print("[green] Conversation cleared. [/green]")
            return True

        case "/save":
            filename = (arg if arg and arg.endswith(".json") else f"{arg}.json") if arg else None
            filepath = conv_manager.save(filename)
            console.print(f"[green]Saved to: {filepath}[/green]")
            return True

        case "/load":
            if not arg: 
                console.print("[yellow]Usage: /load <filename>[/yellow]")
                return True
            
            filename = arg if arg.endswith(".json") else f"{arg}.json"
            if conv_manager.load(filename):
                console.print(f"[green]Loaded: {filename}[/green]")
                console.print(f"[dim]Messages: {len(conv_manager.history)}[/dim]")
            else:
                console.print(f"[red]Not found: {filename}[/red]")
            return True
        
        case "/list":
            conversations = conv_manager.list_conversations()
            if not conversations:
                console.print(f"[yellow]No conversations found.[/yellow]")
            else:
                table = Table(title = "Saved Conversations")
                table.add_column("Filename", style = "cyan")
                table.add_column("Created", style = "green")
                table.add_column("Messages", justify = "right", style = "magenta")

                for conv in conversations:
                    filename = conv["filename"]
                    if filename == conv_manager.current_file:
                        filename = f"* {filename}" # * indicates active session file

                    table.add_row(
                        filename,
                        conv["created"].split("T")[0], # Just date
                        str(conv["message_count"])
                    )
                
                console.print(table)
                if conv_manager.current_file:
                    console.print(f"\n[dim]* indicates current session file[/dim]")
            return True

        case "/stats":
            stats = conv_manager.get_stats()
            table = Table(title="Conversation Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Messages", str(stats["total_messages"]))
            table.add_row("User Messages", str(stats["user_messages"]))
            table.add_row("Assistant Messages", str(stats["assistant_messages"]))
            table.add_row("Total Characters", f"{stats['total_characters']:,}")
            table.add_row("Estimated Tokens", f"{stats['estimated_tokens']:,}")
            table.add_row("Context Usage", stats["context_usage"])
            table.add_row("Percent Used", stats["percent_used"])
            table.add_row("Session File", stats["session_file"])
            
            console.print(table)
            return True

        case "/history":
            if not conv_manager.history:
                console.print(f"[yellow]No history.[/yellow]")
            else:
                for i, msg in enumerate(conv_manager.history, 1):
                    role_color = "green" if msg["role"] == "user" else "blue"
                    role_label = "You" if msg["role"] == "user" else "Assistant"
                    console.print(f"\n[bold {role_color}]{i}. {role_label}: [/bold {role_color}]")
                    console.print(Panel(msg["content"], border_style = role_color))
            return True
        
        case "/sync":
            if not check_server_status():
                console.print("[red]Server not reachable.[/red]")
                return True
            
            console.print(f"[cyan]Syncing...[/cyan]")
            results = manual_sync_all()

            if results["success"] > 0:
                console.print(f"[green]Synced {results['success']}/{results['total']}[/green]")
            if results["failed"] > 0:
                console.print(f"[yellow]Failed {results['failed']}/{results['total']}[/yellow]")
            return True
        
        case "/server":
            if check_server_status():
                console.print("[green]Server online[/green]")
            else:
                console.print("[red]Server offline[/red]")
            return True
        
        case "/remote":
            conversations = list_server_conversations()
            if not conversations:
                console.print("[yellow]No remote conversations[/yellow]")
                return True

            table = Table(title = "Remote conversations")
            table.add_column('Filename', style = 'cyan')
            table.add_column("Created", style="green")
            table.add_column("Size", style="magenta")
            
            for conv in conversations:
                table.add_row(
                    conv.get("filename", "Unknown"),
                    conv.get("created", "Unknown")[:10],
                    f"{conv.get('size', 0) / 1024:.1f} KB"
                )
            console.print(table)
            return True
        
        case "/pull":
            if not arg:
                console.print("[yellow]Usage: /pull <filename>[/yellow]")
                return True
            
            filename = arg if arg.endswith(".json") else f"{arg}.json"
            console.print(f"[cyan]Downloading {filename}...[/cyan]")
            if download_from_server(filename, str(CONVERSATION_DIR)):
                console.print(f"[green]✓ Downloaded to conversations/{filename}[/green]")
            else:
                console.print("[red]✗ Download failed - check server status with /server[/red]")
            return True
        
        case "/model7":
            if not path_7b:
                console.print("[red]path_7b not set in config.toml[/red]")
                return True
            _current_model_name_reset()
            load_model_once(path_7b, "Qwen2.5-Coder-7B")
            return True

        case "/model14":
            if not path_14b:
                console.print("[red]path_14b not set in config.toml[/red]")
                return True
            _current_model_name_reset()
            load_model_once(path_14b, "Qwen2.5-Coder-14B")
            return True
        
        case "/copy":
            if not _CLIPBOARD_AVAILABLE:
                console.print(
                    f"[red]pyperclip not installed.[/red]\nRun: [bold]pip install pyperclip[/bold]"
                )
                return True
            if not last_assistant_response:
                console.print("[yellow]No response to copy yet.[/yellow]")
                return True
            pyperclip.copy(last_assistant_response)
            console.print("[green]Copied to clipboard.[/green]")
            return True

        case "/temp":
            if not arg:
                console.print(f"[cyan]Current temperature: {current_temperature}[/cyan]\n[dim]Usage: /temp <value>  (e.g. /temp 0.3 for focused, /temp 1.0 for creative)[/dim]")
                return True
            try:
                new_temp = float(arg)
                if not (0.0 <= new_temp <= 2.0):
                    raise ValueError("Temperature out of range.")
                current_temperature = new_temp
                console.print(f"[green]Temperature set to {current_temperature}[/green]")
            except ValueError:
                console.print(f"[red]Invalid value. Use a number between 0.0 and 2.0[/red]")
            return True
        
        case "/retry":
            if not conv_manager.history:
                console.print(f"[yellow]Nothing to retry.[/yellow]")
                return True
            
            if conv_manager.history[-1]["role"] != "assistant":
                console.print(f"[yellow]Last message is not from assistant nothing to retry.[/yellow]")
                return True


            if _current_model_name is None and not load_model_once(
                CONFIG["model"].get("default_path", ""), CONFIG["model"].get("default_name", "Unknown")
            ):
                return True
            assert handler is not None

            conv_manager.history.pop()
            console.print("[cyan]Retrying last response...[/cyan]")

            prompt = conv_manager.build_prompt()
            full_response = ""
            first_token_received = False
            token_count = 0
            ttft = 0.0
            start_time_request = time.perf_counter()
            start_time = 0.0

            default_model_name = _current_model_name

            ctx_tokens = conv_manager.count_tokens(
                "".join(m["content"] for m in conv_manager.history)
            )

            def retry_stream_callback(token: str) -> None:
                nonlocal full_response, first_token_received, token_count, start_time, start_time_request, ttft
                
                if not first_token_received:
                    ttft = time.perf_counter() - start_time_request
                    first_token_received = True
                    start_time = time.perf_counter()
                
                token_count += 1
                full_response += token

                elapsed = time.perf_counter() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0

                retry_live.update(
                    Panel(
                        Markdown(full_response),
                        title=f"[bold cyan]{default_model_name}[/bold cyan] | "
                              f"[bold gold1]{tps:.1f} tok/s[/bold gold1] | "
                              f"[dim]ctx {ctx_tokens}/{conv_manager.max_tokens}[/dim]",
                        subtitle=f"[dim]TTFT: {ttft:.3f}s | temp: {current_temperature}[/dim]",
                        border_style="cyan",
                    )
                )
            with Live(console = console, refresh_per_second = 20) as retry_live:
                stop_reason = handler.generate(prompt, retry_stream_callback, current_temperature)
                if stop_reason == "context_full":
                    console.print(f"\n[red]Response cut off. Try /clear or ask more concisely.[/red]")
                if stop_reason == "length_limit":
                    console.print(f"\n[yellow]Reply hit the token limit. Ask it to continue for the rest.[/yellow]")
            
            conv_manager.add_message("assistant", full_response)
            last_assistant_response = full_response

            elapsed = time.perf_counter() - start_time if start_time > 0 else 0
            tps = token_count / elapsed if elapsed > 0 else 0
            console.print(f"\n[dim]Generated {token_count} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)[/dim]")
            return True

        case "/help":
            ctx_tokens = conv_manager.count_tokens(
                "".join(m["content"] for m in conv_manager.history)
            )
            help_text = f"""
[bold cyan]Available Commands:[/bold cyan]

[bold]/clear, /new[/bold]     - Start new conversation
[bold]/save[/bold]            - Save current conversation
[bold]/save <n>[/bold]        - Save with custom name
[bold]/load <n>[/bold]        - Load saved conversation
[bold]/list[/bold]            - List all conversations
[bold]/history[/bold]         - Show conversation history
[bold]/stats[/bold]           - Show statistics
[bold]/sync[/bold]            - Sync to server
[bold]/server[/bold]          - Check server status
[bold]/remote[/bold]          - List remote conversations
[bold]/pull <n>[/bold]        - Download from server
[bold]/help[/bold]            - Show this help
[bold]/model7[/bold]          - Loads qwen 7b, this is the default
[bold]/model14[/bold]         - Loads qwen 14b, note this is intensive, close all apps before loading
[bold]/temp <x>[/bold]        - Set sampling temperature (0.0 to 2.0)
[bold]/retry[/bold]           - Regenerate the last response
[bold]/copy[/bold]            - Copy the last response to the clipboard
[bold]exit, quit, q[/bold]    - Exit program

[dim]End a line wiht \\ for multiline input[/dim]
[dim]Context: {ctx_tokens}/{conv_manager.max_tokens} tokens[/dim]
"""
            console.print(Panel(help_text, title="Help", border_style="cyan"))
            return True
        
        case _:
            # Unknown command
            return False

def _current_model_name_reset():
    global _current_model_name
    _current_model_name = None

def print_gradient_welcome():
    """
    Renders ASCII banner for terminal.
    
    credit: ASCII art created using https://github.com/TheZoraiz/ascii-image-converter
    """
    
    ascii_art = [
"      .::-:.                   . . .  . ",
"    -#@@@@@@@*:              :@@@@@@@@- ",
"  .#@@@@@@@@@@@*             #@@@@@@@=  ",
"  #@@%#*+*#@@@@@#           =@@@@@@@-   ",
" -@@=      .+@@@@=         .@@@@@@@-    ",
" *@-         =@@@@.        *@@@@@@-     ",
" %@           +@@@+       -@@@@@@-      ",
" -:            %@@@       %@@@@@-       ",
"               -@@@-     +@@@@@-        ",
"                %@@#    .@@@@@-         ",
"                -@@@.   #@@@@-          ",
"                 %@@=  -@@@@-           ",
"                 +@@# .@@@@-            ",
"                 .@@@ +@@@-             ",
"                  #@@+@@@-              ",
"                  -@@@@@:               ",
"                  .@@@@:                ",
"                   *@@:                 ",
"                   #@%                  ",
"                  #@@@:                 ",
"                 *@@@@-                 ",
"                =@@@@@=                 ",
"               .@@@@@@=                 ",
"               *@@@@@@=                 ",
"              .@@@@@@@-                 ",
"              :@@@@@@@.                 ",
"              .@@@@@@+                  ",
"               +@@@@+                   ",
"                .::.",
" *✲☆⋆*✧ ✰ ｡* offline llm *✲☆⋆*✧ ✰ ｡*"
]

    for line in ascii_art:
        gradient_text = Text()

        for char in line:
            style = Style(color="white", bold=True)
            gradient_text.append(char, style=style)

        console.print(gradient_text)


def main():
    global last_assistant_response
    # global handler
    # Define a glowing color palette for boarder
    # GLOW_COLORS = [
    #     "bright_blue",
    #     "cyan",
    #     "bright_cyan",
    #     "white",
    #     "bright_white",
    #     "cyan",
    #     "bright_blue",
    #     "blue",
    # ]
    # color_idx = 0
    console.clear()
    print_gradient_welcome()

    default_model_path = CONFIG["model"].get("default_path", "")
    default_model_name = CONFIG["model"].get("default_name", "Unknown")


    try:
        while True:

            user_input = get_multiline_input()

            if user_input.lower() in ["exit", "quit", "q"]:
                # Auto-save on exit if there's conversation history
                if conv_manager.history:
                    save_prompt = Prompt.ask(
                        "[yellow]Save conversation before exiting?[/yellow]",
                        choices = ["y", "n"],
                        default = "y"
                    )
                    if save_prompt.lower() == "y":
                        filepath = conv_manager.save()
                        console.print(f"[green]Saved to {filepath}[/green]")
                break
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                if handle_command(user_input):
                    continue
                console.print(f"[yellow]Unknown command: {user_input.split()[0]}[/yellow]")
                continue
            # Load model

            if _current_model_name is None and not load_model_once(default_model_path, default_model_name):
                continue
            assert handler is not None
            
            # Set or update system prompt
            if not conv_manager._prompt_locked:
                conv_manager.set_system_prompt(build_system_prompt(user_input))

            # Add user message to history
            conv_manager.add_message("user", user_input)

            # Build full prompt with conversation history
            prompt = conv_manager.build_prompt()

            full_response = ""
            first_token_received = False
            token_count = 0
            ttft = 0.0
            start_time_request = time.perf_counter()
            start_time = 0

            ctx_tokens = conv_manager.estimate_tokens("".join(msg["content"] for msg in conv_manager.history))
            ctx_info = f"{ctx_tokens}/{conv_manager.max_tokens} tokens"

            def stream_callback(token: str):
                nonlocal full_response, first_token_received, token_count, start_time, start_time_request, ttft

                if not first_token_received:
                    ttft = time.perf_counter() - start_time_request
                    first_token_received = True
                    start_time = time.perf_counter()

                token_count += 1
                full_response += token

                elapsed_time = time.perf_counter() - start_time
                tps = token_count / elapsed_time if elapsed_time > 0 else 0

                live.update(
                    Panel(
                        Markdown(full_response),
                        title = f"[bold cyan]{_current_model_name}[/bold cyan] | [dim]Context: {ctx_info}[/dim]",
                        subtitle = f"[bold magenta]{tps:.1f} tokens/sec[/bold magenta] | [dim]TTFT: {ttft:.3f}s[/dim]",
                        border_style = "cyan",
                    )
                )

            with Live(console = console, refresh_per_second=20) as live:
                # live.update(Panel(progress, title=m_name, border_style="bright_magenta"))
                # start_time_request = time.perf_counter()
                
                try:
                    stop_reason = handler.generate(prompt, stream_callback, current_temperature)
                except KeyboardInterrupt:
                    stop_reason = "stopped"
                    console.print("\n[yellow]Generation stopped.[/yellow]")

                if stop_reason == "length_limit":
                    # The reply reached the backend's per-reply cap (1024 tokens). Trimming the history
                    # would not make room for a longer reply, so keep the history and what was written.
                    console.print("\n[yellow]Reply hit the token limit. Ask it to continue for the rest.[/yellow]")

                elif stop_reason == "context_full":
                    console.print("\n[yellow]Response cut off detected. Reducing context and resending.[/yellow]")
                    
                    # Save current state
                
                    filepath = conv_manager.save()
                    console.print(f"[dim]Saved to: {filepath}[/dim]")
                    
                    # Trim history to last 2 messages
                    if len(conv_manager.history) > 3:
                        conv_manager.history = conv_manager.history[-3:]
                        console.print(f"[dim]Trimmed to last {len(conv_manager.history)} messages[/dim]")
                    
                    # Rebuild prompt
                    prompt = conv_manager.build_prompt()
                    ctx_tokens = conv_manager.estimate_tokens("".join(msg["content"] for msg in conv_manager.history))
                    ctx_info = f"{ctx_tokens}/{conv_manager.max_tokens} tokens"
                    
                    # Reset tracking
                    full_response = ""
                    first_token_received = False
                    token_count = 0
                    start_time_request = time.perf_counter()

                    console.print("[cyan]Retrying generation...[/cyan]\n")

                    retry_reason = handler.generate(prompt, stream_callback, current_temperature)
                    if retry_reason == "context_full":
                        console.print("\n[red]Still cut off after retry. \nTry clear first.[/red]")
                    elif retry_reason == "length_limit":
                        console.print("\n[yellow]Reply hit the token limit. Ask it to continue for the rest.[/yellow]")
                # with progress:
                #     task = progress.add_task("Generating...", total = None)

                #     def retry_callback(token):
                #         nonlocal full_response, first_token_received, token_count, ttft, start_time

                #         if not first_token_received:
                #             ttft = time.perf_counter() - start_time_request
                #             first_token_received = True
                #             start_time = time.perf_counter()
                        
                #         full_response += token
                #         token_count += 1

                #         console.print(f"[blue]{token}[/blue]", end = "")
                #         progress.update(task, advance = 1)
                    
                #     retry_stop_reason = handler.generate(prompt, retry_callback)
                #     progress.remove_task(task)

                # Add assistant response to history

            last_assistant_response = full_response
            conv_manager.add_message("assistant", full_response)

            elapsed = time.perf_counter() - start_time if start_time > 0 else 0
            tps = token_count / elapsed if elapsed > 0 else 0

            console.print(f"\n[dim]Generated {token_count} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)[/dim]")

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Interrupted...[/yellow]")
    finally:
        # In Rust, dropping the ModelManager handles cleanup automatically.
        console.print("[bold green]Cleanup complete.[/bold green]")


if __name__ == "__main__":
    """
    This script starts the offline LLM. 
    """
    main()