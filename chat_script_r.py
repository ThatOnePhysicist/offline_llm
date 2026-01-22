import model_handler  # Rust lib.rs
import os
import time
from rich.color import Color
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.style import Style
from rich.text import Text

"""
Chat Script Frontend for Offline LLM Inference.

This module provides a Rich-based terminal interface that communicates with a 
Rust backend (model_handler) to run GGUF models locally using Metal acceleration.
It supports dynamic model swapping between 'Coding' and 'Reasoning' models
based on the user's input.
"""

console = Console()

# Local path configs.
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    base_path, "..", "models", "qwen2.5-coder-7b-instruct-q5_k_m.gguf"
)

# Global state for the model engine 
handler = None
_current_model_name = None


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
        f"[bold yellow]Rust:[/bold yellow] Initializing engine for {model_name}..."
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

    handler.load_model(model_name, target_path)

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


def build_prompt(user_input, model_type):
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
        system = "You are a careful, analytical assistant. Explain things step by step."
    else:
        lower = user_input.lower()
        if "rust" in lower:
            system = "You are an experienced Rust developer. Write correct, idiomatic Rust code that compiles."
        elif "kotlin" in lower:
            system = "You are a professional Kotlin developer. Use idiomatic Kotlin patterns and null safety."
        elif "vim" in lower:
            system = "You are a Vim expert. Give minimal, correct answers."
        elif "bash" in lower or "zsh" in lower:
            system = "You are a Unix shell expert. Write safe, POSIX-compliant shell scripts."
        else:
            system = "You are a senior Python engineer. Write clean, idiomatic Python."

    return qwen_instruct(system, user_input)






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

    try:
        while True:
            user_input = console.input("\n[bold green]User: [/bold green]")

            if user_input.lower() in ["exit", "quit", "q"]:
                break

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
            prompt = build_prompt(user_input, model_type)

            full_response = ""
            first_token_received = False
            token_count = 0
            start_time = 0

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None, pulse_style="bright_blue"),
                console=console,
                transient=True,
            )

            progress.add_task("Loading...", total=None)

            def stream_callback(token: str):
                nonlocal \
                    full_response, \
                    first_token_received, \
                    token_count, \
                    start_time, \
                    color_idx

                if not first_token_received:
                    progress.stop()
                    start_time = time.perf_counter()
                    first_token_received = True

                token_count += 1
                full_response += token

                # color_idx = (token_count // 5) % len(GLOW_COLORS)
                # current_border_color = GLOW_COLORS[color_idx]

                elapsed_time = time.perf_counter() - start_time

                tps = token_count / elapsed_time if elapsed_time > 0 else 0

                live.update(
                    Panel(
                        Markdown(full_response),
                        title=f"[bold cyan]{m_name}[/bold cyan]\n[bold gold]{tps:.1f} tokens/sec[/bold gold]",
                        subtitle=f"[bold magenta]{tps:.1f} tokens/sec[/bold magenta]",
                        border_style = "cyan",
                    )
                )

            console.print(f"[bold blue] Assitant ({m_name}): [/bold blue]")
            # progress.start()

            with Live(console=console, refresh_per_second=20) as live:
                live.update(Panel(progress, title=m_name, border_style="bright_magenta"))
                handler.generate(prompt, stream_callback)
                live.update(Panel(Markdown(full_response), title=f"[bold blue]{m_name}[/bold blue]", border_style="bright_green"))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted...[/yellow]")
    finally:
        # In Rust, dropping the ModelManager handles cleanup automatically.
        console.print("[bold green]Rust cleanup complete.[/bold green]")


if __name__ == "__main__":
    """
    This script starts the offline LLM. 
    """
    main()

    