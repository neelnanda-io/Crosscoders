# %%
from utils import *
print(model.cfg.n_layers)
print(cfg)
scratch_cfg = {
    "l1_coeff": 5,
    "dec_init_norm": 0.01,
    "log_every": 5,
    "seed": 50,
    "num_tokens": int(1e7),
}
cfg.update(scratch_cfg)
train = Trainer(cfg, model, use_wandb=False)
# train.total_steps = 10
# train.train()
# %%
cc = CrossCoder.load("version_3", 17, train.model)
buffer = train.buffer
print(buffer.buffer.norm(dim=-1).mean(0))
# %%
# ave_norms = buffer.buffer.norm(dim=-1).mean(0) / np.sqrt(model.cfg.d_model)
ave_norms = torch.tensor(
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
# Mean centered normalisation
mean_norms = torch.tensor([1.4248, 1.5720, 1.6795, 1.8498, 2.0202, 2.2450, 2.5181, 2.9152, 3.3975, 4.1135, 5.1676, 8.3306], device="cuda:0", dtype=torch.float32)
acts2 = buffer.next()
acts = acts2 / ave_norms[:, None] * mean_norms[:, None]
print(acts.shape)
print(acts.norm(dim=-1).mean(0))
print((buffer.buffer[:200] / ave_norms[None, :, None]).norm(dim=-1).mean(0))
# %%
means = buffer.buffer.mean(0) / ave_norms[:, None]
means2 = buffer.buffer.mean(0) / mean_norms[:, None]
line(means)
# %%
cc2 = CrossCoder.load("version_12", 1, train.model)
# %%
torch.set_grad_enabled(False)

recons_acts2 = cc2(acts2.float())
recons_acts = cc(acts.float())
variance = (acts.float() - means.float()).pow(2).sum(-1).mean(0)
variance2 = (acts2.float() - means2.float()).pow(2).sum(-1).mean(0)
mse = (acts.float() - recons_acts).pow(2).sum(-1).mean(0)
mse2 = (acts2.float() - recons_acts2).pow(2).sum(-1).mean(0)
print(mse)
print(mse2)
print(variance)
print(mse.sum() / variance.sum())
line([mse / variance, mse2 / variance], line_labels=["lambda=2", "lambda=5"], title="FVU")


# %%
cc_acts = cc.encode(acts.float())
(cc_acts > 0).float().sum(-1).mean(0)
# %%
model = train.model
batch_size = 128
seq_len = 64
tokens = all_tokens[:batch_size, :seq_len]
with torch.autocast("cuda", torch.bfloat16):
    _, cache = model.run_with_cache(tokens, names_filter=lambda x: x.endswith("resid_post"))
resids = cache.stack_activation("resid_post")
resids = einops.rearrange(resids, "layer batch seq d_model -> (batch seq) layer d_model")

print(resids.shape)
recons_resids = cc(resids / ave_norms[:, None, None]) * ave_norms[None, :, None]
recons_resids2 = cc2(resids / mean_norms[:, None, None]) * mean_norms[None, :, None]
def replace_resids_hook(resid, hook, layer, lambd=2):
    # flat_resid = resid.reshape(-1, model.cfg.d_model)
    # if lambd == 2:
    #     rec = recons_resids[:, layer]
    # else:
    #     rec = recons_resids2[:, layer]
    # diff = flat_resid - rec
    # print(layer, lambd)
    # print(diff.pow(2).sum(-1).mean(0) / flat_resid.pow(2).sum(-1).mean(0))
    # print(diff.pow(2).sum(-1).mean(0), flat_resid.pow(2).sum(-1).mean(0), recons_resids.pow(2).sum(-1).mean(0))

    new_resid = torch.zeros_like(resid)
    new_resid[:, 0, :] = resid[:, 0, :]
    if lambd == 2:
        new_resid[:, 1:, :] = recons_resids[:, layer].reshape(batch_size, seq_len, model.cfg.d_model)[:, 1:, :]
    else:
        new_resid[:, 1:, :] = recons_resids2[:, layer].reshape(batch_size, seq_len, model.cfg.d_model)[:, 1:, :]
    return new_resid
losses = []
orig_loss = model(tokens, return_type="loss")
print(orig_loss)
for lambd in [2, "2_64K"]:
    for layer in tqdm.trange(model.cfg.n_layers):
        loss = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", partial(replace_resids_hook, layer=layer, lambd=lambd))], return_type="loss")
        losses.append({"loss": loss, "lambd": lambd, "layer": layer, "ce_delta": loss - orig_loss})
loss_df = pd.DataFrame(losses)
px.line(loss_df, x="layer", y="ce_delta", color="lambd", title="CE Delta for each layer").show()
print(loss_df.groupby("lambd")["ce_delta"].mean())


# line(losses)
# %%
import sae_lens
import yaml
# with open("/workspace/SAELens/sae_lens/pretrained_saes.yaml", "r") as file:
#     pretrained_saes = yaml.safe_load(file)
# print(pretrained_saes.keys())
RELEASE = "gpt2-small-resid-post-v5-32k"
saes = []
for i in range(model.cfg.n_layers):
    sae = sae_lens.SAE.from_pretrained(
        release=RELEASE,
        sae_id=f"blocks.{i}.hook_resid_post",
        device="cuda",
    )[0]
    saes.append(sae)
# saes.append(
#     sae_lens.SAE.from_pretrained(
#         release=RELEASE,
#         sae_id=f"blocks.11.hook_resid_post",
#         device="cuda",
#     )
# )
# %%
def LN(
    x: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


def replace_resids_sae_hook(resid, hook, layer):
    normed_resid, mu, std = LN(resid[:, 1:, :])
    recons_resid = saes[layer](normed_resid) * std + mu
    resid[:, 1:, :] = recons_resid
    return resid


# losses = []
# orig_loss = model(tokens, return_type="loss")
# print(orig_loss)

for layer in tqdm.trange(model.cfg.n_layers):
    loss = model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (
                f"blocks.{layer}.hook_resid_post",
                partial(replace_resids_sae_hook, layer=layer),
            )
        ],
        return_type="loss",
    )
    losses.append(
        {"loss": loss, "lambd": -1, "layer": layer, "ce_delta": loss - orig_loss}
    )
loss_df = pd.DataFrame(losses)
px.line(
    loss_df, x="layer", y="ce_delta", color="lambd", title="CE Delta for each layer"
).show()
print(loss_df.groupby("lambd")["ce_delta"].mean())
# %%
l0s = []
for i in range(model.cfg.n_layers):
    acts = saes[i].encode(LN(cache["resid_post", i][:, 1:, :])[0])
    l0s.append((acts > 0).sum(-1).float().mean())
fig = line(l0s, return_fig=True, title="L0s for SAEs vs Crosscoders")
filt_resids = resids.reshape(batch_size, seq_len, model.cfg.n_layers, model.cfg.d_model)[:, 1:, :, :]
filt_resids = einops.rearrange(filt_resids, "batch seq layer d_model -> (batch seq) layer d_model")
lambda2l0 = (cc.encode(filt_resids)>0).float().sum(-1).mean()
lambda5l0 = (cc2.encode(filt_resids)>0).float().sum(-1).mean()
fig.add_hline(y=lambda2l0, line_dash="dash", line_color="red", annotation_text="Crosscoder lambda=2")
fig.add_hline(y=lambda5l0, line_dash="dash", line_color="green", annotation_text="Crosscoder lambda=5")
fig.show()
# width = 65
# layer = 18
# %%
loss_df["L0"] = [lambda2l0.item()]*12 + [lambda5l0.item()]*12 + [i.item() for i in l0s]
px.scatter(loss_df, x="L0", y="ce_delta", color="lambd", title="CE Delta vs L0", color_continuous_scale="Portland")

# %%
acts = buffer.next()
recons_acts = cc(acts)
recons_acts2 = cc2(acts)
recons_acts_sae = torch.zeros_like(acts)
for i in range(model.cfg.n_layers):
    normed_acts, mu, std = LN(acts[:, i, :])
    recons_acts_sae[:, i, :] = (saes[i](normed_acts)) * std + mu
means = acts.mean(0)
variance = (acts - means).pow(2).sum(-1).mean(0)
mse = (acts - recons_acts).pow(2).sum(-1).mean(0)
mse2 = (acts - recons_acts2).pow(2).sum(-1).mean(0)
mse_sae = (acts - recons_acts_sae).pow(2).sum(-1).mean(0)
print(mse)
print(mse2)
print(mse_sae)
print(variance)
print(mse.sum() / variance.sum())
line(
    [mse / variance, mse2 / variance, mse_sae / variance], 
    line_labels=["lambda=2", "lambda=5", "SAE"], 
    title="FVU"
)
loss_df["mse"] = to_numpy(torch.cat([mse, mse2, mse_sae]))
loss_df["fvu"] = to_numpy(torch.cat([mse/variance, mse2/variance, mse_sae/variance]))
# %%
loss_df["is_sae"] = loss_df["lambd"] == -1
# Create a mapping for lambda values to marker symbols
lambda_symbols = {2: 'circle', 5: 'square', -1: 'diamond'}

# Create the scatter plot
fig = px.scatter(
    loss_df, 
    x="L0", 
    y="ce_delta", 
    facet_col="layer",
    facet_col_wrap=3,
    symbol="lambd",
    symbol_map=lambda_symbols,
    title="CE Delta vs L0",
    color="is_sae",
    labels={"L0": "L0", "ce_delta": "CE Delta", "layers": "Layer", "lambd": "Lambda"},
    color_continuous_scale="Portland",
    height=1000

)
fig.show()
# %%
x = all_tokens[:256, :64]
_, cache = model.run_with_cache(x, names_filter=lambda x: x.endswith("resid_post"), return_type=None)
# layer, batch, seq, d_model
resids = cache.stack_activation("resid_post")[:, :, 1:, :]
recons_acts_sae = torch.zeros_like(resids)
for i in range(model.cfg.n_layers):
    normed_resids, mu, std = LN(resids[i, :, :])
    recons_acts_sae[i, :, :] = (saes[i](normed_resids)) * std + mu
normed_resids_cc = resids / ave_norms[:, None, None, None]
normed_resids_cc = einops.rearrange(normed_resids_cc, "layer batch seq d_model -> (batch seq) layer d_model")
recons_acts_cc = cc(normed_resids_cc)
recons_acts_cc = einops.rearrange(recons_acts_cc * ave_norms[None, :, None], "(batch seq) layer d_model -> layer batch seq d_model", batch=x.shape[0])
recons_acts_cc2 = cc2(normed_resids_cc)
recons_acts_cc2 = einops.rearrange(recons_acts_cc2 * ave_norms[None, :, None], "(batch seq) layer d_model -> layer batch seq d_model", batch=x.shape[0])
mean_resids = resids.mean([1, 2], keepdim=True)
variance = ((resids - mean_resids) ** 2).sum(-1, keepdim=True).mean([1, 2], keepdim=True)
fvu_sae = ((resids - recons_acts_sae).pow(2)/variance).sum(-1).mean(1)
fvu_cc = ((resids - recons_acts_cc).pow(2)/variance).sum(-1).mean(1)
fvu_cc2 = ((resids - recons_acts_cc2).pow(2)/variance).sum(-1).mean(1)
line(fvu_sae, title="SAE")
line(fvu_cc, title="lambda=2")
line(fvu_cc2, title="lambda=5")
# %%
y = torch.stack([fvu_sae, fvu_cc, fvu_cc2])
line(y, line_labels=["SAE", "lambda=2", "lambda=5"]*20, title="FVU", facet_col=1, facet_col_wrap=3)

line(y.mean(-1), line_labels=["SAE", "lambda=2", "lambda=5"]*20, title="FVU")

# %%
all_sae_acts = []
all_sae_acts2 = []
for _ in tqdm.trange(50):
    acts2 = buffer.next()
    acts = acts2 / ave_norms[:, None] * mean_norms[:, None]
    all_sae_acts.append(cc.encode(acts))
    all_sae_acts2.append(cc2.encode(acts2))
all_sae_acts = torch.cat(all_sae_acts)
all_sae_acts2 = torch.cat(all_sae_acts2)
print(all_sae_acts.shape)
print(all_sae_acts2.shape)
freqs = (all_sae_acts > 0).float().mean(0)
freqs2 = (all_sae_acts2 > 0).float().mean(0)
histogram((freqs+1e-6).log10(), title="SAE")
histogram((freqs2+1e-6).log10(), title="SAE2")
# %%
