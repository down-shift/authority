# Reference: compute resources and their usage rules

**Type:** reference. **Audience:** every agent claiming a `[GPU…]` step.

These rules come from the human (Jerzy) and the machine owners. They are
**non-negotiable**: breaking one risks losing access mid-project. See
`docs/AUTONOMY.md` § Hardware-aware claiming for how they interact with the loop.
Last updated 2026-10-08 (human rules restated; occupancy facts verified that day).

## Which machine to claim (priority order, human rule 2026-10-08)

1. **A100** — if a GPU there is vacant, take it.
2. **RTX PCs** (`gpubox`, `NSSLabPC`) — take any time they are idle, for
   smaller tasks (≤ 8B models in bf16, ≤ 14B quantized; CPU-heavy jobs).
3. **H100** (d.dgx GPU #7) — for big tasks (27B–70B single-GPU runs).
4. **V100** (a.dgx / b.dgx) — for any task, whenever convenient. Also the
   place for **long-lived storage**.

Always check occupancy (`nvidia-smi`) right before launching. Never touch
other people's processes, except where a rule below explicitly allows it.

## A100 cluster — `ssh A100-ya`

- One shared host (`ossiriand`), 8× A100-PCIE-40GB, 7 TB disk (~93% full).
  The ssh alias is in the human's `~/.ssh/config`; `BatchMode=yes` works.
- **Only free GPUs.** Run anything only on GPUs with no other process and
  ~0 MiB used. Pin with `CUDA_VISIBLE_DEVICES`. On 2026-10-08, 6 and 7 were free.
- **No Docker for our user** (socket permission denied), **no uv** and **no HF
  token** preinstalled. Jobs run as plain processes in the home directory:
  per-user uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`), a venv,
  `nohup`/`tmux`. Pass `HF_TOKEN` for gated models (Llama, Gemma) through
  the environment of the job only; never write it into a tracked file.
- 40 GB per card: a ≥27B bf16 model needs two free cards (`tensor_parallel_size=2`
  in vLLM, or `device_map`).
- **Clean up when done:** delete model caches, venvs, and run directories you
  created, after copying the distilled results back. Disk is shared and nearly full.

## RTX workstations — `ssh gpubox`, `ssh NSSLabPC`

| alias | GPU | CPU / RAM / disk | OS / tooling | notes |
|---|---|---|---|---|
| `ssh gpubox` | RTX 5080, 16 GB | 28 cores / 30 GB / ~74 GB free | Ubuntu, docker, git, **no uv** | user `danya`; HF token in `~/.cache/huggingface/token`; holds `~/authority` with the pilot `outputs/` (incl. the Gemma Stage-2 run Phase 0 needs) |
| `ssh NSSLabPC` | RTX 4080 SUPER, 16 GB | 28 cores / 31 GB / ~94 GB free | Arch, docker, git, **no uv** | user `jerzy`; no HF token |

- Both sit on `10.64.8.x`, so **the VPN must be up** (otherwise you get
  `Network is unreachable`). The aliases are in the human's `~/.ssh/config`.
- Use them whenever they're idle; there's no queue. Good for ≤ 8B models in
  bf16, quantized ≤ 14B, dataset builds, tests, and CPU-heavy work. Not for
  27B+ models.
- Workflow: clone the repo into the home directory, install uv per user,
  run with `nohup`/`tmux`, and copy only reports and distilled numbers back.

## H100 cluster — `ssh d.dgx` (host `h100-01`)

- **Only GPU #7** (H100 PCIe 80 GB). Pin it explicitly: `device_ids: ["7"]` in
  compose, and/or `CUDA_VISIBLE_DEVICES` inside the container. The other GPUs
  are off-limits whatever their occupancy.
- **Docker compose only.** Everything runs in compose-managed containers
  (`docker compose up -d` / `down`), launched by the agent over ssh. **Nothing
  is installed natively** on the host: no pip, no uv, no conda, no apt.
- **Files only under `/data/storage/kaluzhnaya_jhub/`.** Compose files, bind
  mounts, the HF cache, datasets, and run outputs all go there and nowhere else
  (not `~`, not `/tmp`). Use a per-task subdirectory, e.g.
  `/data/storage/kaluzhnaya_jhub/authinv/<step-id>/`. Root disk is ~92% full.
- **Other containers:** don't touch them. **Exception:** a container that
  occupies GPU #7 may be stopped to free it. Check which container holds GPU 7
  before stopping anything, and stop only that one.
- **Clean up when done:** `docker compose down`, remove the images you pulled
  if they're not reused soon, and delete your subdirectory once the results
  are copied off. Disk space is limited.
- *Superseded 2026-10-08:* "the human runs the compose file" — agents now run
  compose themselves.

## V100 cluster — docker contexts `adgx` / `bdgx`

- Reached **only through the docker contexts**: `docker --context adgx …` /
  `docker --context bdgx …`. There is no ssh.
- GPU flags differ: **a.dgx needs `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<ids>`**
  (`--gpus` fails there); b.dgx accepts `--gpus`. **b.dgx's driver is 530**, so use
  CUDA ≤ 12.1 images there (12.4 images fail). a.dgx has driver 580.
- 8× V100-SXM2-32GB per node; use **vacant** GPUs only (several can be joined
  over NVLink). **fp16 only**: V100 has no bf16 (`docs/RESEARCH_PLAN.md`
  precision tables refer to H100/A100).
- Storage: both nodes mount `/var/dockerstorage/docker-jkaminskii-202512171636`
  (1 TB, ~93% full on 2026-10-08). Bind-mount it into containers. **Anything may
  be stored there indefinitely**: model weights, caches, frozen datasets, run
  archives. This is the long-term home for artifacts copied off the other machines.

## `.env` handling (human rule, 2026-10-08)

- Agents may **list variable names** from `.env`, never values. The harness
  denies shell reads of `.env`, so use a script that loads it with
  `python-dotenv` and prints names only.
- This project has a $0 API budget (`docs/RESEARCH_PLAN.md`), so no
  OpenRouter or other paid-API keys are used here.

**How to apply:** pick the machine by the priority list, check occupancy, run,
copy results back, archive heavy artifacts to V100 storage if they're worth
keeping, then clean up A100 and H100. See [[decision-autonomy-loop]] for
claiming and releasing steps.
