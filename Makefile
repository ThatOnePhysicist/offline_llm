PYTHON_ENV = ./ai_env/bin/python
CARGO = cargo
LIB_NAME = model_handler
DYLIB_EXT = dylib
SO_EXT = so

TARGET_DIR = ./target/release
LIB_OUT = $(LIB_NAME).$(SO_EXT)

MODEL ?= ../models/DeepSeek-Prover-V2-7B-Q4_K_M.gguf
PNP_REPO ?= $(HOME)/pequalnp-proofs

.PHONY: all clean worker test

# Continuous Lean prover (see README.md). Example: make worker MODEL=/path/to/model.gguf
worker: build
	caffeinate -is $(PYTHON_ENV) worker.py --model $(MODEL) --repo $(PNP_REPO)

test:
	$(PYTHON_ENV) -m unittest discover -s tests -v

# Default: build and run
all: setup build run

setup: ai_env/.pip_installed

ai_env/.pip_installed: requirements.txt
	@echo "Installing Python dependencies..."
	$(PYTHON_ENV) -m pip install -r requirements.txt
	@touch ai_env/.pip_installed

build: $(LIB_OUT)

$(LIB_OUT): src/lib.rs Cargo.toml
	@echo "Building Rust extension..."
	# .cargo/config.toml for macOS dynamic lookup
	@mkdir -p .cargo
	@if [ ! -f .cargo/config.toml ]; then \
		echo '[target.aarch64-apple-darwin]\nrustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]' > .cargo/config.toml; \
	fi
	$(CARGO) build --release
	@echo "Linking .$(DYLIB_EXT) to .$(SO_EXT)..."
	cp $(TARGET_DIR)/lib$(LIB_NAME).$(DYLIB_EXT) $(LIB_OUT)

run: 
	@echo "Starting offline llm..."
	$(PYTHON_ENV) chat_script_r.py

clean:
	@echo "Cleaning up..."
	rm -rf target
	rm -f $(LIB_OUT)
	@echo "Done."

setup:
	@echo "Installing Python dependencies... Please wait..."
	$(PYTHON_ENV) -m pip install -r requirements.txt
	@echo "Setup complete. Run 'make' to start."