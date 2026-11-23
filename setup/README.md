# Setup scripts

- `setup_all.sh`: full first-time setup entrypoint
- `install_system_deps.sh`: Ubuntu/Debian system packages (sudo required)
- `install_miniforge.sh`: per-user Miniforge installation
- `check_host.sh`: NVIDIA driver/GPU and MME path preflight
- `clone_methods.sh`: clone upstream method repositories and record commits
- `make_envs.sh`: create isolated Python environments and install pinned dependencies
- `install_flash_attn.sh`: optional FlashAttention 2 build

Run all scripts from the repository root using `bash setup/setup_all.sh`.
