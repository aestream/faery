(installation)=
# Installation

The easiest way to install Faery is through the [`pypi` reposity using `pip`](https://pypi.org/project/faery):

```shell
pip install faery
```

Note that we recommend the use of [a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to install Faery.


## Using `pipx` or `uv`

Instead of `pip`, you can use alternatives like `pipx` or `uv`.
The installation should be straight-forward

:::{note} Installation via `pipx`
```shell
pipx install faery
```
:::

:::{note} Installation via `uv`
```shell
uv add faery
```
:::


## Installing from source

You can install Faery yourself by pulling the git repository and building the code locally.

```shell
git clone https://github.com/aestream/faery
cd faery
pip install -e .
```

More information about development setup is available in the page about [developing Faery](#development).