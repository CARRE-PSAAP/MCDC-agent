# MCDC Agent

> This agent is in testing. Generated scripts may require manual review.
> Please let Gunnar on slack know if you have any issues.

An AI agent for building [MC/DC](https://github.com/CEMeNT-PSAAP/MCDC) neutron transport simulations. The default experience now uses the v2 pipeline, which provides OpenRouter-backed script generation, interactive learning/Q&A, execution, diagnostics, and visualization.

## Setup

### 1. Install

**Option A: User Install (Recommended)**
Install directly from GitHub without cloning:

```bash
pip install "git+https://github.com/CARRE-PSAAP/MCDC-agent.git"
```

**Option B: Clone and Install**

1. Clone the repository:
   ```bash
   git clone https://github.com/CARRE-PSAAP/MCDC-agent.git
   cd MCDC-agent
   ```
2. Install:
   ```bash
   pip install .
   ```

### 2. Configure OpenRouter

The default v2 experience uses OpenRouter.

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Optionally set a default model:

```bash
export OPENROUTER_MODEL="google/gemini-3-flash-preview"
```

### 3. Legacy Providers (Optional)

The legacy backend is still available explicitly with `--backend legacy`.

For Gemini:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

For Ollama:

```bash
export OLLAMA_MODEL="qwen3:14b"
```

## Usage

The agent provides a CLI command `mcdc-agent`.

### Interactive Mode

The default interactive experience uses the v2 app:

```bash
mcdc-agent interactive
```

### Input Script Generation

Generate a script directly from a prompt with the default v2 backend:

```bash
mcdc-agent generate "[Simulation description]"
```

### Common Commands

- `mcdc-agent generate --file prompt.txt -o run.py`
- `mcdc-agent generate --provider openrouter --model anthropic/claude-opus-4.6 --file prompt.txt`
- `mcdc-agent interactive --file prompt.txt`
- `mcdc-agent generate --backend legacy --provider gemini --file prompt.txt`
- `mcdc-agent --help`
