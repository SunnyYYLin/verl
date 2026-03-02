from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_model = "/vepfs-mlp2/mlp-public/zhongcuiting/models/HybriDNA-300M-instruct"
out_model  = "/vepfs-mlp2/mlp-public/zhongcuiting/models/HybriDNA-300M-instruct-ext"

# 你新增的 cell types（显式写出来，避免歧义）
new_label_tokens = ["[BJAB]", "[THP-1]", "[Jurkat]", "[GM12878]"]

# 1. load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
)

print("Original vocab size:", tokenizer.vocab_size)

# 2. 检查这些 token 是否已存在
existing = set(tokenizer.get_vocab().keys())
tokens_to_add = [t for t in new_label_tokens if t not in existing]

print("Tokens to add:", tokens_to_add)

# 3. 如果 tokenizer 还没包含这些 token，才 add
if len(tokens_to_add) > 0:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens_to_add}
    )

print("New vocab size:", tokenizer.vocab_size)

# ========= 关键检查点 1 =========
assert all(t in tokenizer.get_vocab() for t in new_label_tokens)

# 4. load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

old_n = model.get_input_embeddings().weight.shape[0]
new_n = tokenizer.vocab_size

print(f"Resize embedding: {old_n} → {new_n}")

# ========= 关键检查点 2 =========
assert new_n >= old_n

# 5. resize embedding
if new_n > old_n:
    model.resize_token_embeddings(new_n)

    # 初始化新 token embedding（安全做法）
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[old_n:] = emb[:old_n].mean(dim=0)

# ========= 关键检查点 3 =========
assert model.get_input_embeddings().weight.shape[0] == tokenizer.vocab_size

# 6. 保存
tokenizer.save_pretrained(out_model)
model.save_pretrained(out_model)

print("✅ Done. Extended model saved to:", out_model)
