# Install Instructions

If you want to set up your own staging, environment, run the commands below.  You should not need to do this if you use the GPU environment. It has been initialised with everything you need.  If there is a library not installed, you can install it into your base environment on the GPU server and it will be persistent.

If you have your own GPU, you can replace faiss-cpu to faiss-gpu below.

To be clear, you DO NOT HAVE TO DO THIS ON THE GPU. All of this is in the
base environment. This is just the details that will allow you to reproduce
it on your own computer if you wish.

```bash
conda create -n "COMP4703A2" python=3.9
conda activate COMP4703A2
pip install torch torchvision openai easydict tqdm numpy scikit-learn
pip install matplotlib seaborn llama-index==0.9.40 transformers==4.40.2
pip install jupyterhub notebook faiss-cpu sentencepiece bpe
pip install protobuf FlagEmbedding peft bs4 datasets lightgbm more-itertools
pip install tokenizers sentence-transformers safetensors rich pandas psutil pyarrow
pip install "huggingface_hub[cli]"
```

If the FlagEmbedding install fails, you can try: 
```bash
pip install git+https://github.com/FlagOpen/FlagEmbedding.git
```

Or if all else fails, do you can try:
```bash
git clone https://github.com/FlagOpen/FlagEmbedding.git 
cd FlagEmbedding 
pip install -e .
```

You may need to run "huggingface-cli login" if you use a gated LLM that is not already pre-cached. You will know as you will see an error telling you that you must login in order to download model "XX", where XX is the model you set in # the ranker, reranker, or rag example.


