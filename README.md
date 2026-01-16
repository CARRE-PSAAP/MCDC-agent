# MCDC Agent

> [!WARNING]
> This agent is in testing. Generated scripts may require manual review.

An AI agent that guides you through building [MC/DC](https://github.com/CEMeNT-PSAAP/MCDC) neutron transport simulations. Uses Google Gemini to help you generate simulation input scripts. Currently does not support the various MCDC techniques like `population_control`, `branchless_collision()`, etc., they will be added in the future.


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

### 2. Configure Gemini API
You need a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

```bash
export GEMINI_API_KEY="your-api-key-here"
```

If you encounter rate limits, you can set up billing (it will cost a few cents) or switch models by setting the `GEMINI_MODEL` environment variable:

```bash
export GEMINI_MODEL="gemini-2.5-flash"  # or "gemini-2-flash"
```

## Usage

The agent provides a CLI command `mcdc-agent`.

### Interactive Mode
Best for beginners. The agent guides you step-by-step.

```bash
mcdc-agent interactive
```

### Input Script Generation (Recommended)
Generate a script directly from a prompt.

```bash
mcdc-agent generate "[Simulation description]"
```

### Options
- `mcdc-agent generate --file prompt.txt -o run.py`: Read from file and save output.
- `mcdc-agent --help`: Show all available commands.
