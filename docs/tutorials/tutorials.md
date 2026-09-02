# Tutorials

Step-by-step guides that demonstrate how to use faery - both standalone and together with other tools/libraries like [PyTorch](https://pytorch.com) and [JAX](https://jax.dev).
Each tutorial is a Jupyter notebook stored as [jupytext](https://jupytext.readthedocs.io/) Markdown: read it here with rendered outputs, or download the source from the [docs/tutorials directory](https://github.com/aestream/faery/tree/main/docs/tutorials) and run it yourself.
You can either open the `.md` file directly in Jupyter Lab (with the jupytext extension installed) or convert it first:

```sh
pip install jupytext
jupytext --to ipynb <tutorial>.md
```

| Tutorial | What it covers | Extra requirements |
|----------|----------------|--------------------|
| [Streaming into PyTorch & JAX](stream_to_pytorch.md) | Fixed-rate packets, DLPack frame and sparse export, zero-copy tensor import, a streaming convolution | `torch`, `jax` |

Shorter, self-contained scripts live in the
[examples directory](https://github.com/aestream/faery/tree/main/examples).
