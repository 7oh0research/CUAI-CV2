# DECISIONS
- PyramidDrop 단계 비율 / lambda=(R/576)^(1/3), 층 [8,16,24] / 마지막 단계 토큰 수를 R 로 맞추기 위함
- VisionZip dominant:contextual / 0.84375:0.15625 / 공식 설정 54+10=64 의 비율
- batch 마지막 불완전 배치 / 버림 / 모든 방법에 같은 배치 구성
- prefill 정의 / generate 호출 시작 ~ 첫 토큰 생성 직후 / vision+select+LLM prefill 포함, CPU 전처리 제외
