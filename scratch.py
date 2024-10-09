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
train = Trainer(cfg, model)
# train.total_steps = 10
train.train()
# %%
acts = train.buffer.next()
acts.shape
cc = train.crosscoder
print(cc.W_enc.shape)
print(cc.W_dec.shape)
print(cc.W_enc.norm(dim=0).mean(-1))
print(cc.W_dec.norm(dim=-1).mean(0))
encoded_acts = cc.encoder(acts)
print(encoded_acts.shape)
decoded_acts = cc.decoder(encoded_acts)
print(decoded_acts.shape)
# %%
