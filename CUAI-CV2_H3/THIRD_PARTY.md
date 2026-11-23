# Third-party sources and assets

This project orchestrates but does not relicense the following upstream projects:

- LLaVA: https://github.com/haotian-liu/LLaVA
- FastV: https://github.com/pkunlp-icler/fastv (confirm canonical URL/revision before use)
- SparseVLM: https://github.com/Gumpest/SparseVLMs
- VisionZip: https://github.com/JIA-Lab-research/VisionZip

`setup/clone_methods.sh` uses the canonical upstream URLs where known. Check each repository's current license, dependencies, and recommended commit before running. Keep a record of exact commit hashes in `results/reports/source_revisions.txt`.

Model weights and MME/POPE data are not bundled. Download them only from their official distribution pages and follow their terms.
