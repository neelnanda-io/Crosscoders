# %%
import torch
from utils import *
from transformer_lens import *

sd = torch.load("/workspace/Crosscoders/checkpoints/version_2/17.pt")
model = HookedTransformer.from_pretrained("gpt2-small")

normalisation_factor = torch.tensor(
    [
        1.8281,
        2.0781,
        2.2031,
        2.4062,
        2.5781,
        2.8281,
        3.1562,
        3.6875,
        4.3125,
        5.4062,
        7.8750,
        16.5000,
    ],
    device="cuda:0",
    dtype=torch.float32,
)
# %%


def fold_scale_into_sd(sd, normalisation_factor):
    d = {}
    print(sd.keys())
    d["W_enc"] = sd["W_enc"] / normalisation_factor[:, None, None]
    d["W_dec"] = sd["W_dec"] * normalisation_factor[None, :, None]
    d["b_enc"] = sd["b_enc"]
    d["b_dec"] = sd["b_dec"] * normalisation_factor[:, None]
    return d


d = fold_scale_into_sd(sd, normalisation_factor)
for k in sd.keys():
    print(k, sd[k].shape, d[k].shape)

# %%
cfg = json.load(open("/workspace/Crosscoders/checkpoints/version_2/17_cfg.json", "r"))
cc = CrossCoder(cfg, model)
cc_normed = CrossCoder(cfg, model)
cc_normed.load_state_dict(d)
# %%
data = load_dataset("stas/openwebtext-10k")
# %%
s = data["train"][0]["text"]
tokens = model.to_tokens(s)
print(tokens.shape)
# %%
resids = get_stacked_resids(model, tokens, drop_bos=True)
resids.shape
# %%
resids = resids.squeeze(0)
recons = cc(resids)
recons_normed = cc_normed(resids)
print(resids.norm(), (resids - recons).norm(), (resids - recons_normed).norm())
# %%
cfg = json.load(open("/workspace/Crosscoders/checkpoints/version_12/1_cfg.json", "r"))
sd = torch.load("/workspace/Crosscoders/checkpoints/version_12/1.pt")
sd_normed = fold_scale_into_sd(sd, normalisation_factor)
name = "lambda_2_64k"
json.dump(cfg, open(f"/workspace/Crosscoders/checkpoints/{name}_cfg.json", "w"))
torch.save(sd_normed, f"/workspace/Crosscoders/checkpoints/{name}.pt")
# %%
cc_new = CrossCoder.load(name, model)
print(cc_new.get_losses(resids))
print(cc.get_losses(resids))
print(cc_normed.get_losses(resids))
# %%
import huggingface_hub

upload_folder_to_hf(f"/workspace/Crosscoders/checkpoints/", "crosscoders-gpt2-small")

# %%
