# MLAPP-Solutions

Solutions and thoughts on Kevin Murphy's *Machine Learning: a Probabilistic Perspective*, fourth printing.

## Directories

- `tex`: LaTeX source code and pdfs with solutions and comments to exercises.
- `src`: Implementations of models
- `notebooks`: Demos of models.

## Installation

First, switch to Python 3.8 (I use pyenv):

```bash
pyenv shell 3.8.5
```

Install `poetry`[https://python-poetry.org] if it isn't present:

```bash
pip install poetry
```

Finally, create a virtual environment, install dependencies, and add the venv to the kernel:

```bash
python -m venv .venv
source .venv/bin/activate
poetry install
python -m ipykernel install --user --name=mlapp_solutions
```

## Testing

Test can be run with

```bash
make tests
```

If you want more detailed coverage, use

```bash
make html
```
