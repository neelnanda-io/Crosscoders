from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crosscoders",
    version="0.1.0",
    author="Neel Nanda",
    author_email="neel@neelnanda.io",
    description="A package for training GPT-2 Small acausal crosscoders",
    long_description=long_description,
    long_description_content_type="markdown",
    url="https://github.com/neelnanda-io/CrossCoders",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformer_lens",
        "wandb",
        "einops",
        "tqdm",
        "numpy",
        "huggingface_hub",
    ],
)