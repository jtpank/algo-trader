# Algo pipeline

Explanation goes here.

## Setup

The pipeline is implemented in python and can be installed using poetry.
To install poetry (OSX, Linux, or WSL) run:

```
curl -sSL https://install.python-poetry.org | python3 -
```

Then from the `./algo-trader` directory run the command below to install dependencies:

```bash
poetry install
```

Once installed you can run the multi-step pipeline, or run each step individually.

Running Scripts:

```bash
poetry run python main.py
```
## Development

Please use `black` and `isort` to format all python scripts. Run:
```
poetry run black .
poetry run isort .
```
## Tests

Implement all tests in the `./tests` directory following the pytest framework.
```
poetry run pytest
```

