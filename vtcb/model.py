"""LLaVA-1.5 runner with visual-token pruning.

Design choice (important for reviewers): this repo measures ACCURACY, not latency.
  * Pre-LLM methods (random, pool2d, visionzip, divprune) really remove tokens: the LLM
    receives K image embeddings. Positions are renumbered contiguously, as in VisionZip.
  * In-LLM methods (fastv, pdrop, dart) are implemented by MASKING pruned image tokens as
    keys in every layer after the pruning point (prefill and decoding). Pruned tokens keep
    being computed but no kept token can read them, so the output is the same as physically
    dropping them while keeping original position ids (FastV's reference behaviour).
    This needs eager attention so that a 4D mask is materialized; the code asserts it.
Do not report latency from this repo.

Simplifications vs. the original papers are listed in METHODS.md.
"""
import math, hashlib
import torch, torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

IMG_TOKENS = 576
GRID = 24

def _parts(model):
    """Return (vision_tower, projector, causal_lm, decoder_layers) across transformers versions."""
    vt, proj = model.vision_tower, model.multi_modal_projector
    lm = model.language_model
    if hasattr(model, "lm_head"):            # >=4.52 layout: language_model is the bare LlamaModel
        layers = lm.layers
        return vt, proj, None, layers
    return vt, proj, lm, lm.model.layers     # 4.51 layout: LlamaForCausalLM

def seed_for(item_id):
    return int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16)


class Pruner:
    """State shared by the attention hooks during one generation."""
    def __init__(self):
        self.reset(None, None, None)

    def reset(self, method, budget, cfg, img_start=None):
        self.method, self.budget, self.cfg = method, budget, cfg or {}
        self.img_start = img_start
        self.removed = {}       # layer -> LongTensor of absolute positions masked from that layer on
        self.stages = []        # [(layer, keep_count)] filled per method
        self.query_pos = None   # absolute position of the last prompt token
        self.log = {}

    # ---- schedule ---------------------------------------------------------
    def plan(self):
        c, B = self.cfg, self.budget
        if self.method in ("fastv", "dart"):
            self.stages = [(c["prune_after_layer"] + 1, B)]
        elif self.method == "pdrop":
            self.stages = [(l + 1, max(1, round(B * f)))
                           for l, f in zip(c["prune_after_layers"], c["keep_frac_of_budget"])]
        else:
            self.stages = []

    def masked_at(self, layer):
        out = [v for l, v in self.removed.items() if l <= layer]
        return torch.cat(out) if out else None


class LlavaPruned:
    def __init__(self, cfg):
        self.cfg = cfg
        dt = getattr(torch, cfg["dtype"])
        self.proc = AutoProcessor.from_pretrained(cfg["model"])
        self.model = LlavaForConditionalGeneration.from_pretrained(
            cfg["model"], torch_dtype=dt, attn_implementation="eager").cuda().eval()
        self.vt, self.proj, self.lm, self.layers = _parts(self.model)
        self.embed = self.model.get_input_embeddings()
        self.tok = self.proc.tokenizer
        self.P = Pruner()
        self._attn_cache = {}
        for i, layer in enumerate(self.layers):
            layer.self_attn.register_forward_pre_hook(self._pre_hook(i), with_kwargs=True)
            layer.self_attn.register_forward_hook(self._post_hook(i), with_kwargs=True)
        self.vsl = self.model.config.vision_feature_layer               # -2
        self.strategy = self.model.config.vision_feature_select_strategy  # "default" -> drop CLS

    # ---------------- hooks for in-LLM methods ------------------------------
    def _pre_hook(self, i):
        def hook(mod, args, kwargs):
            P = self.P
            if not P.stages:
                return None
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            prefill = hs.shape[1] > 1
            # selection at stage layers (prefill only)
            for (l, keep) in P.stages:
                if l == i and prefill and l not in P.removed:
                    P.removed[l] = self._select_in_llm(i, keep, hs)
            pos = P.masked_at(i)
            if pos is None:
                return None
            m = kwargs.get("attention_mask")
            assert m is not None and m.dim() == 4, \
                "in-LLM pruning needs an explicit 4D mask (eager attention)"
            m = m.clone()
            m[..., pos.to(m.device)] = torch.finfo(m.dtype).min
            kwargs["attention_mask"] = m
            return args, kwargs
        return hook

    def _post_hook(self, i):
        def hook(mod, args, kwargs, out):
            P = self.P
            if P.method in ("fastv", "pdrop") and any(l - 1 == i for l, _ in P.stages):
                w = out[1]
                if w is not None and w.shape[2] > 1:      # prefill attention probs
                    self._attn_cache[i] = w[0, :, P.query_pos, :].float().mean(0)  # (kv,)
            return None
        return hook

    def _select_in_llm(self, layer, keep, hs):
        """Return absolute positions to REMOVE from `layer` on (among still-alive image tokens)."""
        P = self.P
        img = torch.arange(P.img_start, P.img_start + IMG_TOKENS, device=hs.device)
        dead = P.masked_at(layer - 1)
        alive = img if dead is None else img[~torch.isin(img, dead.to(hs.device))]
        keep = min(keep, alive.numel())
        if P.method in ("fastv", "pdrop"):
            a = self._attn_cache.pop(layer - 1, None)
            assert a is not None, f"no attention captured at layer {layer-1}"
            score = a[alive]
            kept = alive[score.topk(keep).indices]
        elif P.method == "dart":
            x = hs[0, alive].float()
            n_piv = min(P.cfg["n_pivots"], keep)
            piv = x.norm(dim=-1).topk(n_piv).indices
            xn = F.normalize(x, dim=-1)
            sim = (xn @ xn[piv].T).max(-1).values      # redundancy w.r.t. pivots
            sim[piv] = -2.0                             # pivots always kept
            kept = alive[sim.topk(keep, largest=False).indices]
        else:
            raise ValueError(P.method)
        removed = alive[~torch.isin(alive, kept)]
        P.log[f"kept@{layer}"] = int(kept.numel())
        return removed

    # ---------------- pre-LLM methods ---------------------------------------
    @torch.no_grad()
    def _visual(self, image, method, budget):
        """-> projected image embeddings (K, d)."""
        pv = self.proc.image_processor(image, return_tensors="pt")["pixel_values"]
        pv = pv.to(self.model.device, self.model.dtype)
        need_attn = method == "visionzip"
        out = self.vt(pv, output_hidden_states=True, output_attentions=need_attn)
        feat = out.hidden_states[self.vsl]                 # (1, 577, 1024)
        assert self.strategy == "default"
        x = feat[0, 1:]                                    # drop CLS -> (576, 1024)
        if method in (None, "vanilla", "fastv", "pdrop", "dart") or budget >= IMG_TOKENS:
            sel = x
        elif method == "random":
            g = torch.Generator().manual_seed(self._rseed)
            idx = torch.randperm(IMG_TOKENS, generator=g)[:budget].sort().values
            sel = x[idx.to(x.device)]
        elif method == "pool2d":
            s = int(math.isqrt(budget)); assert s * s == budget, "pool2d needs a square budget"
            g = x.T.reshape(1, -1, GRID, GRID).float()
            sel = F.adaptive_avg_pool2d(g, s).reshape(x.shape[1], -1).T.to(x.dtype)
        elif method == "visionzip":
            sel = self._visionzip(x, out.attentions[self.vsl], budget)
        elif method == "divprune":
            sel = x[self._divprune(x, budget)]
        else:
            raise ValueError(method)
        return self.proj(sel.unsqueeze(0))[0]

    def _visionzip(self, x, attn, budget):
        D = max(1, round(budget * self.cfg["methods"]["visionzip"]["dominant_frac"]))
        C = budget - D
        cls_att = attn[0, :, 0, 1:].float().mean(0)        # CLS -> patches, (576,)
        dom = cls_att.topk(D).indices.sort().values
        rest = torch.ones(IMG_TOKENS, dtype=torch.bool, device=x.device); rest[dom] = False
        r = x[rest].float()
        if C == 0:
            return x[dom]
        step = max(1, r.shape[0] // C)
        tgt_idx = torch.arange(0, r.shape[0], step, device=x.device)[:C]
        is_tgt = torch.zeros(r.shape[0], dtype=torch.bool, device=x.device); is_tgt[tgt_idx] = True
        tgt, oth = r[is_tgt], r[~is_tgt]
        assign = (F.normalize(oth, dim=-1) @ F.normalize(tgt, dim=-1).T).argmax(-1)
        summ = tgt.clone(); cnt = torch.ones(C, device=x.device)
        summ.index_add_(0, assign, oth); cnt.index_add_(0, assign, torch.ones_like(assign, dtype=cnt.dtype))
        ctx = (summ / cnt[:, None]).to(x.dtype)
        return torch.cat([x[dom], ctx], 0)

    @staticmethod
    def _divprune(x, budget):
        xn = F.normalize(x.float(), dim=-1)
        dist = 1 - xn @ xn.T                                 # cosine distance
        dist.fill_diagonal_(float("inf"))
        first = dist.min(1).values.argmax()
        sel = [first.item()]
        mind = dist[first].clone(); mind[first] = -1
        for _ in range(budget - 1):
            j = mind.argmax().item(); sel.append(j)
            mind = torch.minimum(mind, dist[j]); mind[sel] = -1
        return torch.tensor(sorted(sel), device=x.device)

    # ---------------- prompt + generation ------------------------------------
    def _embed_text(self, s, bos):
        ids = self.tok(s, add_special_tokens=False, return_tensors="pt").input_ids
        if bos:
            ids = torch.cat([torch.tensor([[self.tok.bos_token_id]]), ids], 1)
        return self.embed(ids.to(self.model.device))[0]

    @torch.no_grad()
    def generate(self, item, method="vanilla", budget=IMG_TOKENS, control="image"):
        """control in {image, blank, noimage}; method/budget apply only to control=image."""
        mcfg = self.cfg["methods"].get(method, {})
        self._rseed = seed_for(item["id"]) ^ self.cfg["seed"]
        q = item["question"]
        if control == "noimage":
            emb = self._embed_text(f"USER: {q} ASSISTANT:", bos=True)
            img_start, n_img = None, 0
        else:
            image = item["image"]
            if control == "blank":
                mean = tuple(int(255 * m) for m in self.proc.image_processor.image_mean)
                image = Image.new("RGB", image.size, mean)
            v = self._visual(image, method if control == "image" else "vanilla",
                             budget if control == "image" else IMG_TOKENS)
            pre = self._embed_text("USER: ", bos=True)
            post = self._embed_text(f"\n{q} ASSISTANT:", bos=False)
            emb = torch.cat([pre, v.to(pre.dtype), post], 0)
            img_start, n_img = pre.shape[0], v.shape[0]
        in_llm = control == "image" and method in ("fastv", "pdrop", "dart") and budget < IMG_TOKENS
        self.P.reset(method if in_llm else None, budget, mcfg, img_start)
        self.P.query_pos = emb.shape[0] - 1
        self.P.plan()
        self._attn_cache.clear()
        text = self._greedy(emb.unsqueeze(0), item["max_new_tokens"], need_attn=in_llm and method != "dart")
        info = {"n_img_tokens_in": n_img, **self.P.log}
        self.P.reset(None, None, None)
        return text, info

    def _greedy(self, emb, max_new, need_attn):
        lm = self.lm if self.lm is not None else self.model
        L = emb.shape[1]
        mask = torch.ones(1, L, dtype=torch.long, device=emb.device)
        out = lm(inputs_embeds=emb, attention_mask=mask, use_cache=True, output_attentions=need_attn)
        past, ids = out.past_key_values, []
        nxt = out.logits[0, -1].argmax()
        for _ in range(max_new):
            if nxt.item() == self.tok.eos_token_id:
                break
            ids.append(nxt.item())
            mask = torch.cat([mask, mask.new_ones(1, 1)], 1)
            out = lm(input_ids=nxt.view(1, 1), attention_mask=mask, past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[0, -1].argmax()
        return self.tok.decode(ids, skip_special_tokens=True).strip()
