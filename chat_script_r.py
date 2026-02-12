from typing import Any

from rich import prompt


import model_handler  # Rust lib.rs
import os
import time
import json
from datetime import datetime
from pathlib import Path
from rich.color import Color
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

console = Console()

# Local path configs
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    base_path, "..", "models", "qwen2.5-coder-7b-instruct-q5_k_m.gguf"
)

# Conversation storage path
CONVERSATION_DIR = Path(base_path) / "conversations"
CONVERSATION_DIR.mkdir(exist_ok = True)

# Global state for the model engine 
handler = None
_current_model_name = None

# Conversation state
conversation_history = []
current_system_prompt = ""
MAX_CONTEXT_TOKENS = 3700

class ConversationManager:
    """Manages conversation history, context window, and persistence."""

    def __init__(self, max_tokens = MAX_CONTEXT_TOKENS) -> None:
        self.history = []
        self.system_prompt = ""
        self.max_tokens = MAX_CONTEXT_TOKENS
        self.current_file = None

    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.
        """
        self.history.append(
            {
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
            console.print(f"\n[dim]Keeping last 2 exchanges for continuty with current message[/dim]")

            self.history.append(new_message)

    def set_system_prompt(self, prompt: str):
        """
        Set system prompt.
        """
        self.system_prompt = prompt
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rought token estimation (approx. 1 token ~ 4 chars in English).
        Considering this is just for offline use, this should be okay. 
        If necessary, will look into using a tokenizer. 
        """
        return len(text) // 4
    
    def is_near_limit(self, threshold = 0.90) -> bool:
        """
        Checks if approaching context limit.

        Parameters
        ----------
        threshold : float
            Trigger limit [default] 90%.
        """
        current_tokens = self.estimate_tokens("".join(msg["content"] for msg in self.history))
        return current_tokens >= (self.max_tokens * threshold)

    def get_trimmed_history(self) -> list:
        """
        Returns conversation history trimmed to fit within context window.
        Keeps most recent messages and ensures sytem prompt is always included.
        """
        system_tokens = self.estimate_tokens(self.system_prompt)
        available_tokens = self.max_tokens - system_tokens

        trimmed = []
        token_count = 0

        for msg in reversed(self.history):
            msg_tokens = self.estimate_tokens(msg["content"])
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
        if clear_file_tracking:
            self.current_file = None

    def save(self, filename: str = None) -> str:
        """
        Save conversation to JSON file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        filepath = CONVERSATION_DIR / filename

        data = {
            "created": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "history": self.history,
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.current_file:
            filename = self.current_file
        else:
            filename = f"conversation_{timestamp}"

        filepath = self.save(filename)

        if keep_last_n > 0:
            keep_count = keep_last_n * 2
            kept_messages = self.history[-keep_count::] if len(self.history) > keep_count else self.history
            self.history = kept_messages
        else:
            self.history = []
        
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

        return True

    def list_conversations(self) -> list:
        """
        List all saved conversations.
        """
        conversations = []
        for filepath in CONVERSATION_DIR.glob("conversation_*.json"):
            with open(filepath, "r", encoding = "utf-8") as f:
                data = json.load(f)
            conversations.append(
                {
                    "filename": filepath.name,
                    "created": data.get("created", "Unknown"),
                    "message_count": len(data.get("history", []))
                }
            )
        
        return sorted(conversations, key = lambda x: x["created"], reverse = True)

    def get_stats(self) -> dict:
        """
        Get conversation statistics.
        """
        total_messages = len(self.history)
        user_messages = sum(1 for msg in self.history if msg["role"] == "user")
        assistant_messages = sum(1 for msg in self.history if msg["role"] == "assistant")

        total_chars = sum(len(msg["content"]) for msg in self.history)
        estimated_tokens = self.estimate_tokens("".join(msg["content"] for msg in self.history))

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

def clean_output(text: str) -> str:
    """
    Removes chat-specific tokens from the model input.
    
    Parameters
    ----------
    text : str
        Raw string from tokenizer.
    
    Returns
    -------
    str 
        Cleaned text for terminal.    
    """

    for token in ("<|im_end|>", "<|im_start|>"):
        text = text.replace(token, "")
    return text.strip()


def load_model_once(model_id: str, model_name: str):
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
        return

    if handler is None:
        handler = model_handler.ModelManager()

    console.print(
        f"[bold yellow] Initializing engine for {model_name}...[/bold yellow]"
    )

    try:
        if not os.path.exists(model_id):
            raise FileNotFoundError(f"File not found at: {model_id}")

        # Rust reads GGUF and maps tensors
        result = handler.load_model(model_name, model_id)
        console.print(f"[green]{result}[/green]")
        _current_model_name = model_name

    except Exception as e:
        console.print(f"[bold red]Load Error:[/bold red] {e}")
        raise e

    target_path = model_id

    # handler.load_model(model_name, target_path)

    _current_model_name = model_name
    Console().print(f"[dim]Model {model_name} loaded in Metal VRAM.[/dim]")


def qwen_instruct(system_prompt, user_prompt):
    """
    Formats the input required by Qwen2.5/DeepSeek.
    
    Parameters
    ----------
    system_prompt : str
        Instructions for the model.
    user_prompt : str
        User query.
    """
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

def choose_model(user_input):
    """
    Simple router for model choice, 'code' model is default.

    Parameters
    ----------
    user_input : str
        User query.
    """
    keywords = [
        "explain",
        "why",
        "reason",
        "prove",
        "step by step",
        "analysis",
        "derive",
    ]
    if any(k in user_input.lower() for k in keywords):
        return "reason"
    return "code"


def build_system_prompt(user_input, model_type):
    """
    Constructs the system prompt based on language based keywords or 'reason'.

    Parameters
    ----------
    user_input : str
        User query.
    model_type : str
        Model type to use determined by `choose_model` function. 
    """
    if model_type == "reason":
        return "You are a careful, analytical assistant. Explain things step by step."
    else:
        lower = user_input.lower()
        if "rust" in lower:
            return "You are an experienced Rust developer. Write correct, idiomatic Rust code that compiles."
        elif "kotlin" in lower:
            return "You are a professional Kotlin developer. Use idiomatic Kotlin patterns and null safety."
        elif "vim" in lower:
            return "You are a Vim expert. Give minimal, correct answers."
        elif "bash" in lower or "zsh" in lower:
            return "You are a Unix shell expert. Write safe, POSIX-compliant shell scripts."
        else:
            return "You are a senior Python engineer. Write clean, idiomatic Python."

    # return qwen_instruct(system, user_input)

def handle_command(command: str) -> bool:
    """
    Handle special commands. Returns True if command was handled.
    """
    cmd = command.lower().strip()

    if cmd in ["/clear", "/new"]:
        conv_manager.clear(clear_file_tracking = True)
        console.print("[green] Conversation cleared. [/green]")
        return True

    elif cmd == "/save":
        filepath = conv_manager.save()
        console.print(f"[green] Conversation saved to: {filepath}[/green]")
        return True

    elif cmd.startswith("/save "):
        filename = cmd.split(" ", 1)[1].strip()
        if not filename.endswith(".json"):
            filename += ".json"
        filepath = conv_manager.save(filename)
        console.print(f"[green] Conversation saved to: {filepath}[/green]")
        console.print(f"[dim]this file will be used for future auto-saves in this session.[/dim]")
        return True

    elif cmd.startswith("/load "):
        filename = cmd.split(" ", 1)[1].strip()
        if not filename.endswith(".json"):
            filename += ".json"
        if conv_manager.load(filename):
            console.print(f"[green] Conversation loaded.[/green]")
            console.print(f"[dim]Messages: {len(conv_manager.history)}[/dim]")
        else:
            console.print(f"[red] Could not find: {filename}[/red]")
        return True

    elif cmd == "/list":
        conversations = conv_manager.list_conversations()
        if not conversations:
            console.print(f"[yellow]No saved conversations.[/yellow]")
        else:
            table = Table(title = "Saved Conversations")
            table.add_column("Filename", style = "cyan")
            table.add_column("Created", style = "green")
            table.add_column("Messages", justify = "right", style = "magenta")

            for conv in conversations:
                filename = conv["filename"]
                if filename == conv_manager.current_file:
                    filename = f">{filename}" # > indicates active session file

                table.add_row(
                    filename,
                    conv["created"].split("T")[0], # Just date
                    str(conv["message_count"])
                )
            
            console.print(table)
            
            if conv_manager.current_file:
                console.print(f"\n[dim]> indicates current session file[/dim]")
        return True

    elif cmd == "/stats":
        stats = conv_manager.get_stats()
        table = Table(title = "Conversation statistics")
        table.add_column("Metric", style = "cyan")
        table.add_column("Value", style = "green")

        table.add_row("Total Messages", str(stats["total_messages"]))
        table.add_row("User Messages", str(stats["user_messages"]))
        table.add_row("Assistant Messages", str(stats["assistant_messages"]))
        table.add_row("Total Characters", f"{stats['total_characters']:,}")
        table.add_row("Estimated Tokens", f"{stats['estimated_tokens']:,}")
        table.add_row("Context Usage", stats["context_usage"])
        table.add_row("Percent used", stats["percent_used"])
        table.add_row("Session File", stats["session_file"])

        console.print(table)

        return True
    
    elif cmd == "/history":
        if not conv_manager.history:
            console.print("[yellow]No conversation history.[/yellow]")
        else:
            for i, msg in enumerate(conv_manager.history, 1):
                role_color = "green" if msg ["role"] == "user" else "blue"
                role_label = "You" if msg["role"] == "user" else "Assistant"
                
                console.print(f"\n[bold {role_color}]{i}. {role_label}:[/bold {role_color}]")
                console.print(Panel(msg["content"], border_style = role_color))
        return True
    
    elif cmd == "/help":
        help_text = """
[bold cyan]Available commands:[/bold cyan]

[bold]/clear, /new[/bold]       - Start a new conversation.
[bold]/save[/bold]              - Save current conversation (auto-named).
[bold]/save <name>[/bold]       - Save current conversation with a custom name.
[bold]/load <name>[/bold]       - Load a saved conversation.
[bold]/list[/bold]              - List all saved conversations.
[bold]/history[/bold]           - Show current conversation history.
[bold]/stats[/bold]             - Show conversation statistics.
[bold]/help[/bold]              - Show this help message.
[bold]exit, quit, q[/bold]      - Quit chat.

[dim]Context window {}/{} tokens (approx)[/dim]
[dim]Auto-saves at 85% capacity, appends to session file[/dim]
        """.format(
            conv_manager.estimate_tokens("".join(msg["content"] for msg in conv_manager.history)),
            conv_manager.max_tokens
        )
        console.print(Panel(help_text, title = "Help", border_style = "cyan"))
        return True
    
    return False




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
    global handler
    # Define a glowing color palette for boarder
    GLOW_COLORS = [
        "bright_blue",
        "cyan",
        "bright_cyan",
        "white",
        "bright_white",
        "cyan",
        "bright_blue",
        "blue",
    ]
    color_idx = 0
    console.clear()

    abs_deepseek = "/Users/alexanderpenaflor/ai_build/models/DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf"
    abs_qwen_coder = "/Users/alexanderpenaflor/ai_build/models/qwen2.5-coder-7b-instruct-q5_k_m.gguf"
    
    print_gradient_welcome()
    # console.print("\n[dim]Type /help for available commands[/dim]\n")

    try:
        while True:
            user_input = console.input("\n[dim]Type /help for available commands[/dim]\n[bold green]User: [/bold green]")

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
            
            # Handle commands
            if user_input.startswith("/"):
                if handle_command(user_input):
                    continue

            # Choose model based on query
            model_type = choose_model(user_input)

            if model_type == "reason":
                m_path, m_name = (
                    abs_deepseek,
                    "DeepSeek-R1-7B",
                )
            else:
                m_path, m_name = (
                    abs_qwen_coder,
                    "Qwen2.5-Coder-7B",
                )

            # Load model via Rust.
            load_model_once(m_path, m_name)

            # Set or update system prompt
            system_prompt = build_system_prompt(user_input, model_type)
            conv_manager.set_system_prompt(system_prompt)

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

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width = None, pulse_style = "bright_blue"),
                console = console,
                transient = True,
            )

            progress.add_task("Loading...", total = None)

            def stream_callback(token: str):
                nonlocal \
                    full_response, \
                    first_token_received, \
                    token_count, \
                    start_time, \
                    start_time_request, \
                    ttft, \
                    color_idx

                if not first_token_received:
                    ttft = time.perf_counter() - start_time_request
                    # print(f"Time to first token: {ttft:.4f}s")
                # if not first_token_received:
                    start_time = time.perf_counter()
                    first_token_received = True
                    progress.stop()

                token_count += 1
                full_response += token

                # color_idx = (token_count // 5) % len(GLOW_COLORS)
                # current_border_color = GLOW_COLORS[color_idx]

                elapsed_time = time.perf_counter() - start_time
                tps = token_count / elapsed_time if elapsed_time > 0 else 0

                # Show context usage in panel
                ctx_tokens = conv_manager.estimate_tokens("".join(msg["content"] for msg in conv_manager.history))
                ctx_info = f"{ctx_tokens}/{conv_manager.max_tokens} tokens"

                live.update(
                    Panel(
                        Markdown(full_response),
                        title = f"[bold cyan]{m_name}[/bold cyan]\n[bold gold]{tps:.1f} tokens/sec[/bold gold] | [dim]Context: {ctx_info}[/dim]",
                        subtitle = f"[bold magenta]{tps:.1f} tokens/sec[/bold magenta] | [dim]TTFT: {ttft:.3f}s[/dim]",
                        border_style = "cyan",
                    )
                )

            # console.print(f"[bold blue] Assitant ({m_name}): [/bold blue]")
            # progress.start()

            with Live(console = console, refresh_per_second=20) as live:
                # live.update(Panel(progress, title=m_name, border_style="bright_magenta"))
                start_time_request = time.perf_counter()
                handler.generate(prompt, stream_callback)

                # Add assistant response to history
                conv_manager.add_message("assistant", full_response)

                live.update(
                    Panel(
                        Markdown(full_response), 
                        title=f"[bold blue]{m_name}[/bold blue]", 
                        border_style="bright_green"
                        )
                    )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted...[/yellow]")
    finally:
        # In Rust, dropping the ModelManager handles cleanup automatically.
        console.print("[bold green]Cleanup complete.[/bold green]")


if __name__ == "__main__":
    """
    This script starts the offline LLM. 
    """
    main()