"""Mean/last-token embeddings of the window texts with Qwen3-Embedding-0.6B (text baseline of probe-control).
  python scripts/lens/pc_embed.py data/probe_control/sample_texts.json data/probe_control/text_emb.npy
Sample order = pc_select.jsonl order, one sample per (run, cut index) as in pc_analyze.Pairs."""
import sys, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
texts = json.load(open(sys.argv[1]))
name = "Qwen/Qwen3-Embedding-0.6B"
tok = AutoTokenizer.from_pretrained(name, padding_side="left")
m = AutoModel.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
out = []
order = np.argsort([len(t) for t in texts])
for i in range(0, len(texts), 8):
    b = [texts[j] for j in order[i:i + 8]]
    enc = tok(b, padding=True, truncation=True, max_length=2048, return_tensors="pt").to("cuda")
    with torch.no_grad():
        h = m(**enc).last_hidden_state[:, -1]
    out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
E_ = np.concatenate(out); res = np.zeros_like(E_); res[order] = E_
np.save(sys.argv[2], res); print(res.shape)
