import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

class Vlm:
    def __init__(self, model_id, dtype="float16", attn="sdpa"):
        self.dtype = getattr(torch, dtype)
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = "left"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype, attn_implementation=attn).to("cuda").eval()
        self.attn_used = getattr(self.model.config, "_attn_implementation", attn)

    def _prompt(self, text, with_image):
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": text}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}],
                                             add_generation_prompt=True)

    @torch.inference_mode()
    def generate(self, texts, images, max_new_tokens):
        """images=None 이면 이미지 토큰 없이 텍스트만 입력한다."""
        with_image = images is not None
        prompts = [self._prompt(t, with_image) for t in texts]
        if with_image:
            imgs = [im.convert("RGB") for im in images]
            enc = self.proc(images=imgs, text=prompts, return_tensors="pt", padding=True)
        else:
            enc = self.proc(text=prompts, return_tensors="pt", padding=True)
        enc = {k: (v.to("cuda", self.dtype) if v.dtype.is_floating_point else v.to("cuda"))
               for k, v in enc.items()}
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  num_beams=1, pad_token_id=self.proc.tokenizer.pad_token_id
                                  or self.proc.tokenizer.eos_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        return self.proc.batch_decode(new, skip_special_tokens=True)

def blank_like(img, gray=128):
    return Image.new("RGB", img.size, (gray, gray, gray))
