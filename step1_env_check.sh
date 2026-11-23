#!/usr/bin/env bash
set -e
python -m h1bench.env_check
for e in $(conda env list | awk '/^h1_/{print $1}'); do
  conda run -n $e python -c "import json,torch,transformers;d={'torch':torch.__version__,'transformers':transformers.__version__}
try:
  import flash_attn;d['flash_attn']=flash_attn.__version__
except Exception as x:d['flash_attn']=None
print(json.dumps(d))" > env/$e.json || echo "{}" > env/$e.json
done
# flash-attn 이 어떤 env 에도 없으면 flash 조건 전체 unsupported_env
if ! grep -h flash_attn env/h1_*.json | grep -qv null; then echo '{"unsupported_env_flash": true}' > env/env_flags.json; fi
echo "다음: bash step2_smoke_test.sh   (확인 파일: env/env.json)"
