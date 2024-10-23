Code to train a GPT-2 Small acausal crosscoder

Details (written in the context of someone training a model diff crosscoder):
* I based this on my initial replication of Towards Monosemanticity, not any of the SAE libraries (seemed clunkier to adapt those). Not sure this was the right call
* Key files: utils.py has basically everything important, train.py is a tiny file that calls and runs the trainer, and calling it with eg --l1_coeff=2 will update the config used, scratch.py has some analysis code. Ignore all other files as irrelevant
* I decided to implement it with W_enc having shape [n_layers, d_model, d_sae] and W_dec having shape [d_sae, n_layers, d_model] (here you'd change n_layers to two). It'd also be reasonable to implement it by flattening it into a n_layers * d_model axis and just having a funkier loss function that needs to unflatten, but this felt more elegant to me
* I followed the Anthropic April update method and some adaptions from the crosscoder post like the loss function
* I separately computed and hard coded the normalisation factors, I think these are fairly important. Probably less so here since base and chat should have v similar norms(?)
* This is using ReLU and L1 - I expect topK or JumpReLU would just be better (no shrinkage or "needing to have small activations" issues) and basically work fine, though Anthropic did say something about the weird loss (sum of L2 norm of each layer) incentivising layer sparsity, which may be lost with those? It's probably fine to stick with it as is. Gated with their L1 loss variant may also be fine, idk.
* There's a buffer which runs the model on several batches periodically and stores a shuffled mix of activations and provides them. You'll need to adapt this to run both chat and base (ideally have the same control tokens in both so it's perfectly matched, unless this breaks the base model?)
* I store a pre-tokenized dataset locally and just load it as a global tensor called all_tokens. This is very hacky, but should be easy to swap out
* I found that it was very sensitive to the W_dec init norm - I initially made each d_model vector 0.1 and this went terribly. I think the norm of the flattened vector should probably be 0.1? I just fiddled a bit and found something kinda fine
* Probably not relevant to you, but I found that the crosscoder was much better on earlier layers than later (eg 35% FVU on layer 10, <10% on the first few layers)

-----
# This is all old stuff that's probably no longer relevant
# TLDR

This is an open source replication of [Anthropic's Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) paper. The autoencoder was trained on the gelu-1l model in TransformerLens, you can access two trained autoencoders and the model using [this tutorial](https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn#scrollTo=MYrIYDEfBtbL). 

# Reading This Codebase

This is a pretty scrappy training codebase, and won't run from the top. I mostly recommend reading the code and copying snippets. See also [Hoagy Cunningham's Github](https://github.com/HoagyC/sparse_coding).

* `utils.py` contains various utils to define the Autoencoder, data Buffer and training data. 
  * Toggle `loading_data_first_time` to True to load and process the text data used to run the model and generate acts
* `train.py` is a scrappy training script
  * `cfg["remove_rare_dir"]` was an experiment in training an autoencoder whose features were all orthogonal to the shared direction among rare features, those lines of code can be ignored and weren't used for the open source autoencoders. 
  * There was a bug in the code to set the decoder weights to have unit norm - it makes the gradients orthogonal, but I forgot to *also* set the norm to be 1 again after each gradient update (turns out a vector of unit norm plus a perpendicular vector does not remain unit norm!). I think I have now fixed the bug. 
* `analysis.py` is a scrappy set of experiments for exploring the autoencoder. I recommend reading the Colab tutorial instead for something cleaner and better commented. 

Setup Notes:

* Create data - you'll need to set the flag loading_data_first_time to True in utils.py , note that this downloads the training mix of gelu-1l and if using eg the Pythia models you'll need different data (I recommend https://huggingface.co/datasets/monology/pile-uncopyrighted )
* A bunch of folders are hard coded to be /workspace/..., change this for your system.
* Create a checkpoints dir in /workspace/1L-Sparse-Autoencoder/checkpoints

* If you train an autoencoder and want to share the weights, copy the final checkpoints to a new folder, use upload_folder_to_hf to upload to HuggingFace, create your own repo. Run huggingface-cli login to login, and apt-get install git-lfs and then git lfs install
