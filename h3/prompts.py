"""LLaVA-1.5 평가 관행에 맞춘 질문 문구. 이미지 유무와 관계없이 텍스트는 동일하다."""
SHORT = "Answer the question using a single word or phrase."
MC = "Answer with the option's letter from the given choices directly."

def build_text(it):
    if it["type"] in ("yesno", "open"):
        return f"{it['question']}\n{SHORT}"
    lines = []
    if it.get("hint"):
        lines.append(f"Hint: {it['hint']}")
    lines.append(it["question"])
    for L, c in zip(it["letters"], it["choices"]):
        lines.append(f"{L}. {c}")
    lines.append(MC)
    return "\n".join(lines)
