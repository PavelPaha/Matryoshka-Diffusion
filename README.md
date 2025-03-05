To install packages run these commands:

```uv venv --python 3.12```
```source .venv/bin/activate.fish``` or ```source .venv/bin/activate```
``` uv lock```
``` uv sync --all-extras --frozen```


To add env to jupyter kernel run:
```uv run python3 -m ipykernel install --user --name MatryoshkaDiffusion```
