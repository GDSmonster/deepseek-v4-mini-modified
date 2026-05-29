# AGENTS.md

Compact orientation for agents working in this repo. See `README.md` and `docs/` for full context.

## Project shape

Pedagogical from-scratch PyTorch implementation of DeepSeek-V4-style components. Pure Python package, no compiled extensions, no Hydra. Configs are YAML under `config/` but not auto-loaded; high-level APIs take plain Python kwargs / dataclass configs (e.g. `DeepSeekV4LMConfig`).

Top-level packages (each is its own setuptools package, listed in `pyproject.toml`):
- `src/` - model architecture (`mini_deepseek_class.DeepSeekV4LM` is the entrypoint). Note: imports look like `from src.mini_deepseek_class import ...` - `src` is a real package, NOT a layout root.
- `training/`, `inference/`, `data/`, `parallel/`, `ablations/`, `scripts/` - importable as top-level modules.
- `tests/` mirrors these areas plus loose `tests/test_*.py` files for core components.

## Commands

- Install: `pip install -e ".[dev,data]"` (Python >=3.10, torch>=2.2).
- Full test suite: `pytest` (CPU-safe, ~756 passing). CUDA tests auto-skip.
- Targeted: `pytest tests/training`, `pytest tests/data`, `pytest tests/inference`, `pytest tests/parallel`, `pytest tests/experiments` (ablations live here, NOT `tests/ablations`).
- Loose component tests (model-tests CI job) run with `pytest tests/*.py` - this glob skips subdirectories on purpose.
- Lint: `ruff check .` (config in `pyproject.toml`: line-length 100, py310, ignores F403/F405/E501). No separate typecheck.
- Markers: `slow`, `cuda` are declared; `--strict-markers` is on, so any new marker must be registered in `pyproject.toml`.
- CLIs (after editable install): `deepseekv4-{data,train,inspect,parallel,infer,ablate}`. Equivalent `python -m scripts.<name>_cli ...` always works.

## Conventions and gotchas

- Keep tests CPU-safe and tiny (see `CONTRIBUTING.md`); gate GPU code with `pytest.mark.skipif(not torch.cuda.is_available(), ...)` or the `cuda` marker.
- `pyproject.toml` `[tool.pytest.ini_options]` adds both `.` and `src` to `pythonpath`. Don't rename the `src` package or rewrite imports as `from mini_deepseek_class import ...` - existing code uses the `src.` prefix.
- `pyproject.toml` `packages` is an explicit list. When adding a new top-level package or a subpackage of `src/`, update it (currently only `src.transformer_modules` is listed under `src`).
- `training/chekpoints.py` is intentionally misspelled - imports rely on it. Don't "fix" silently.
- CI is path-filtered (`.github/workflows/ci.yml`). Touching `pyproject.toml`, `requirements.txt`, or `.github/workflows/**` triggers all suites; otherwise only the matching folder runs. Mirror this when reasoning about which tests must pass.
- Ablations: tests are under `tests/experiments/`, CLI is `scripts/ablation_cli.py`, outputs land in `outputs/ablations/{id}/`. Runner clears Torch/CUDA cache between variants - don't add global state that survives that.
- Inference cache modes are `audit`, `mha_decode`, `deepseek_decode` with `deepseek_prefill_mode in {parallel, ...}`. The `deepseek_decode` path requires building HCA/CSA layer caches via `inference/deepseek_cache_builder.py`; raw MHA caches are not interchangeable.
- Model-parallel V1: every `balance` entry must be > 0, so `len(devices) <= n_layers`. Wrap with `wrap_model_parallel` BEFORE building the optimizer.
- Don't commit checkpoints, datasets, or notebook outputs (per `CONTRIBUTING.md`).

## Where to look first

- Architecture entrypoint: `src/mini_deepseek_class.py` (`DeepSeekV4LM`, `DeepSeekV4LMConfig`).
- Training entrypoint: `training/train_deepseek.py::train_deepseekv4`.
- Inference entrypoint: `inference/__init__.py` (`inference_autoregresive`).
- Ablation suites: `ablations/suites.py` (`ablation_1` ... `ablation_6`).
- Deep docs: `docs/README.md` indexes architecture, training, inference cache, config reference, and CLI pages.
