# %%
import os

os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
os.environ["DATASETS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *
import wandb
from torch.nn.utils import clip_grad_norm_
import huggingface_hub

# %%
import argparse


def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg


default_cfg = {
    "seed": 51,
    "batch_size": 2048,
    "buffer_mult": 512,
    "lr": 2e-5,
    "num_tokens": int(4e8),
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "dict_size": 2**16,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    # "remove_rare_dir": False,
    "model_name": "gpt2-small",
    "site": "resid_post",
    # "layer": 0,
    "device": "cuda:0",
    "model_batch_size": 32,
    "log_every": 100,
    "save_every": 100000,
    "dec_init_norm": 0.005,
}

cfg = arg_parse_update_cfg(default_cfg)


def post_init_cfg(cfg):
    cfg["name"] = f"{cfg['model_name']}_{cfg['dict_size']}_{cfg['site']}"


post_init_cfg(cfg)
pprint.pprint(cfg)
# %%

SEED = cfg["seed"]
GENERATOR = torch.manual_seed(SEED)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(True)

# model: HookedTransformer = (
#     HookedTransformer.from_pretrained(cfg["model_name"])
#     .to(DTYPES[cfg["enc_dtype"]])
#     .to(cfg["device"])
# )

# n_layers = model.cfg.n_layers
# d_model = model.cfg.d_model
# n_heads = model.cfg.n_heads
# d_head = model.cfg.d_head
# d_mlp = model.cfg.d_mlp
# d_vocab = model.cfg.d_vocab

# %%
# Replace with your own path
SAVE_DIR = Path("/workspace/Crosscoders/checkpoints")

from typing import NamedTuple


class LossOutput(NamedTuple):
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor


class CrossCoder(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])
        self.W_enc = nn.Parameter(
            torch.empty(
                model.cfg.n_layers, model.cfg.d_model, d_hidden, dtype=self.dtype
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    d_hidden, model.cfg.n_layers, model.cfg.d_model, dtype=self.dtype
                )
            )
        )
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec.data = (
            self.W_dec.data
            / self.W_dec.data.norm(dim=-1, keepdim=True)
            * self.cfg["dec_init_norm"]
        )
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "d_hidden n_layers d_model -> n_layers d_model d_hidden",
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((model.cfg.n_layers, model.cfg.d_model), dtype=self.dtype)
        )

        self.d_hidden = d_hidden

        self.to(self.cfg["device"])

        self.save_dir = None
        self.save_version = 0

    def encode(self, x, apply_relu=True):
        # x: [batch, n_layers, d_model]
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "... n_layers d_model, n_layers d_model d_hidden -> ... d_hidden",
        )
        if apply_relu:
            acts = F.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc
        return acts

    def decode(self, acts):
        # acts: [batch, d_hidden]
        acts_dec = einops.einsum(
            acts,
            self.W_dec,
            "... d_hidden, d_hidden n_layers d_model -> ... n_layers d_model",
        )
        return acts_dec + self.b_dec

    def forward(self, x):
        # x: [batch, n_layers, d_model]
        acts = self.encode(x)
        return self.decode(acts)

    def get_losses(self, x):
        # x: [batch, n_layers, d_model]
        x = x.to(self.dtype)
        acts = self.encode(x)
        # acts: [batch, d_hidden]
        x_reconstruct = self.decode(acts)
        diff = x_reconstruct.float() - x.float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(squared_diff, "... n_layers d_model -> ...", "sum")
        l2_loss = l2_per_batch.mean()

        decoder_norms = self.W_dec.norm(dim=-1)
        # decoder_norms: [d_hidden, n_layers]
        total_decoder_norm = einops.reduce(
            decoder_norms, "d_hidden n_layers -> d_hidden", "sum"
        )
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)

        l0_loss = (acts > 0).float().sum(-1).mean()

        return LossOutput(l2_loss=l2_loss, l1_loss=l1_loss, l0_loss=l0_loss)

    def create_save_dir(self):
        version_list = [
            int(file.name.split("_")[1])
            for file in list(SAVE_DIR.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.save_dir = SAVE_DIR / f"version_{version}"
        self.save_dir.mkdir(parents=True)

    def save(self):
        if self.save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        print(f"Saved as version {self.save_version} in {self.save_dir}")
        self.save_version += 1

    @classmethod
    def load(
        cls,
        name,
        model=None,
        path="",
    ):
        # If the files are not in the default save directory, you can specify a path
        # It's assumed that weights are [name].pt and cfg is [name]_cfg.json
        if path == "":
            save_dir = SAVE_DIR
        else:
            save_dir = Path(path)
        cfg_path = save_dir / f"{str(name)}_cfg.json"
        weight_path = save_dir / f"{str(name)}.pt"

        cfg = json.load(open(cfg_path, "r"))
        pprint.pprint(cfg)
        if model is None:
            model = (
                HookedTransformer.from_pretrained(cfg["model_name"])
                .to(DTYPES[cfg["enc_dtype"]])
                .to(cfg["device"])
            )
        self = cls(cfg=cfg, model=model)
        self.load_state_dict(torch.load(weight_path))
        return self

    @classmethod
    def load_from_hf(cls, version):
        """
        Loads the saved autoencoder from HuggingFace.

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version == "run1":
            version = 25
        elif version == "run2":
            version = 47

        cfg = utils.download_file_from_hf(
            "NeelNanda/sparse_autoencoder", f"{version}_cfg.json"
        )
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(
            utils.download_file_from_hf(
                "NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True
            )
        )
        return self


# %%
def get_stacked_resids(model, tokens, drop_bos=True):
    """
    Returns the stacked activations of the resid_post hook for all layers.

    This could be made about 2x more memory efficient with a buffer, but who cares.

    If drop_bos is true, the resids on the BOS (first) token is dropped.

    Returns stacked_resids: [batch, seq_len (- 1), n_layers, d_model]
    """
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda x: x.endswith("resid_post")
    )
    stacked_resids = torch.stack(
        [cache["resid_post", i] for i in range(model.cfg.n_layers)], dim=-2
    )
    if drop_bos:
        stacked_resids = stacked_resids[:, 1:, :, :]
    return stacked_resids


# %%
def shuffle_data(all_tokens):
    print("Shuffling data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]


def push_to_hub(local_dir):
    if isinstance(local_dir, huggingface_hub.Repository):
        local_dir = local_dir.local_dir
    os.system(f"git -C {local_dir} add .")
    os.system(f"git -C {local_dir} commit -m 'Auto Commit'")
    os.system(f"git -C {local_dir} push")


def upload_folder_to_hf(folder_path, repo_name=None, debug=False):
    """
    Uploads a folder to HuggingFace, and creates a repo for it.
    """
    folder_path = Path(folder_path)
    if repo_name is None:
        repo_name = folder_path.name
    repo_folder = folder_path.parent / (folder_path.name + "_repo")
    repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)
    repo = huggingface_hub.Repository(repo_folder, repo_url)

    for file in folder_path.iterdir():
        if debug:
            print(file.name)
        file.rename(repo_folder / file.name)
    push_to_hub(repo.local_dir)


# loading_data_first_time = False
# if loading_data_first_time:
#     raise NotImplementedError("This is not implemented yet")
#     data = load_dataset(
#         "NeelNanda/c4-code-tokenized-2b", split="train", cache_dir="/workspace/cache/"
#     )
#     data.save_to_disk("/workspace/data/c4_code_tokenized_2b.hf")
#     data.set_format(type="torch", columns=["tokens"])
#     all_tokens = data["tokens"]
#     all_tokens.shape

#     all_tokens_reshaped = einops.rearrange(
#         all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128
#     )
#     all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
#     all_tokens_reshaped = all_tokens_reshaped[
#         torch.randperm(all_tokens_reshaped.shape[0])
#     ]
#     torch.save(all_tokens_reshaped, "/workspace/data/c4_code_2b_tokens_reshaped.pt")
# else:
#     # data = datasets.load_from_disk("/workspace/data/c4_code_tokenized_2b.hf")
#     all_tokens = torch.load("/workspace/data/owt_tensor.pt")
#     # all_tokens = all_tokens[: cfg["num_tokens"] // cfg["seq_len"]]
#     # all_tokens = shuffle_data(all_tokens)


# %%
class Buffer:
    """
    This defines a data buffer, to store a stack of acts across all layers that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        self.buffer = torch.zeros(
            (self.buffer_size, model.cfg.n_layers, model.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"])
        self.cfg = cfg
        self.model = model
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        # average norm of residuals per layer / sqrt(d_model)
        # We divide by this to normalise the data. This is *not* mean centered.
        self.normalisation_factor = torch.tensor(
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
        # The factors when mean centering (ie using the standard deviation)
        # self.normalisation_factor = torch.tensor(
        #     [
        #         1.4248,
        #         1.5720,
        #         1.6795,
        #         1.8498,
        #         2.0202,
        #         2.2450,
        #         2.5181,
        #         2.9152,
        #         3.3975,
        #         4.1135,
        #         5.1676,
        #         8.3306,
        #     ],
        #     device=cfg["device"],
        # )
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        print("Refreshing the buffer!")
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.buffer_batches
            else:
                num_batches = self.buffer_batches // 2
            self.first = False
            for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"]):
                tokens = all_tokens[
                    self.token_pointer : min(
                        self.token_pointer + self.cfg["model_batch_size"], num_batches
                    )
                ]
                _, cache = self.model.run_with_cache(
                    tokens, names_filter=lambda x: x.endswith("resid_post")
                )
                cache: ActivationCache

                acts = cache.stack_activation("resid_post")
                acts = acts[:, :, 1:, :]  # Drop BOS
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )

                # print(tokens.shape, acts.shape, self.pointer, self.token_pointer)
                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]
                # if self.token_pointer > all_tokens.shape[0] - self.cfg["model_batch_size"]:
                #     self.token_pointer = 0

        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
        ]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        # out: [batch_size, n_layers, d_model]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            # print("Refreshing the buffer!")
            self.refresh()
        if self.normalize:
            # Make each layer's vector have expected stdev sqrt(d_model).
            # I use stdev not norm because the mean is easy to learn.
            out = out / self.normalisation_factor[None, :, None]
        return out


class Trainer:
    def __init__(self, cfg, model, use_wandb=True):
        self.cfg = cfg
        self.model = model
        self.crosscoder = CrossCoder(cfg, model)
        self.buffer = Buffer(cfg, model)
        self.total_steps = cfg["num_tokens"] // cfg["batch_size"]

        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )
        self.step_counter = 0

        if use_wandb:
            wandb.init(project="crosscoder", entity="neelnanda-io", config=cfg)

    def lr_lambda(self, step):
        if step < 0.05 * self.total_steps:
            return step / (0.05 * self.total_steps)
        elif step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps, then keeps it constant
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def step(self):
        acts = self.buffer.next()
        losses = self.crosscoder.get_losses(acts)
        loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
        }
        self.step_counter += 1
        return loss_dict

    def log(self, loss_dict):
        wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    def save(self):
        self.crosscoder.save()

    def train(self):
        self.step_counter = 0
        try:
            for i in tqdm.trange(self.total_steps):
                loss_dict = self.step()
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()
        finally:
            self.save()


# buffer.refresh()
# %%


# %%
def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post


# %%
# Frequency
@torch.no_grad()
def get_freqs(num_batches=25, local_encoder=None):
    raise NotImplementedError("This is not implemented yet")
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).to(
        cfg["device"]
    )
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[: cfg["model_batch_size"]]]

        _, cache = model.run_with_cache(
            tokens, stop_at_layer=cfg["layer"] + 1, names_filter=cfg["act_name"]
        )
        acts = cache[cfg["act_name"]]
        acts = acts.reshape(-1, cfg["act_size"])

        hidden = local_encoder(acts)[2]

        act_freq_scores += (hidden > 0).sum(0)
        total += hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores == 0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores
