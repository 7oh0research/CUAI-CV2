# 실행 가이드

## 0. 준비
```bash
unzip h1_bench.zip && cd h1_bench
bash setup/clone_methods.sh            # 공식 저장소 clone + commit 기록
export ANCHORPRUNE_PATH=/path/to/anchorprune   # 없으면 생략(blocked 처리)
export MME_ROOT=/path/to/MME_Benchmark_release_version
bash setup/make_envs.sh                # 방법별 conda env 생성
```

## 1~6단계
```bash
bash step1_env_check.sh        # -> env/env.json
bash step2_smoke_test.sh       # -> results/smoke/*.json
bash step3_main_R32_bs1.sh
bash step4_batch_sweep.sh
bash step5_R_sweep.sh
bash step6_aggregate.sh        # -> summary.csv, ratios.csv, figs
```
또는 `bash run_all.sh` (resume 지원, 끝난 조합은 건너뜀).

## 단계별로 확인할 파일
| 단계 | 확인 파일 |
|---|---|
| 1 | env/env.json |
| 2 | results/smoke/*.json, results/smoke/smoke_report.txt (남은 토큰 수 = R 여부) |
| 3~5 | results/raw/*.jsonl, logs/*.log |
| 6 | results/summary.csv, results/ratios.csv, results/figs/ |

smoke_report.txt에서 `MISMATCH`가 나온 방법은 configs/methods.yaml의 해당 키를
공식 저장소 README/스크립트와 대조해서 맞춘 뒤 step2를 다시 실행한다.
