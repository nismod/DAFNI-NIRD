# DAFNI-NIRD

National Infrastructure Resilience Demonstrator (NIRD)

## Development setup

Clone this repository:

    git clone git@github.com:nismod/DAFNI-NIRD.git

(Or, if you prefer to use HTTPS authentication, `git clone https://github.com/nismod/DAFNI-NIRD.git`)

Move into the cloned folder:

    cd DAFNI-NIRD

Create a conda environment using
[micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
to install packages specified in the [`environment.yaml`
file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually):

    micromamba env create -f environment.yaml

Activate it:

    micromamba activate nird

Configure the [pre-commit](https://pre-commit.com/) checks:

    pre-commit install

There are several tools and helpers set up to run automatically, on `git commit`
and in [GitHub Actions](https://docs.github.com/en/actions) continuous
integration steps. Each of these can be run locally too.

Run the tests using [pytest](https://docs.pytest.org):

    python -m pytest

Run formatting using [black](https://black.readthedocs.io/):

    black .

Run linting using [ruff](https://docs.astral.sh/ruff/):

    ruff check .

Run type-checking using [mypy](https://mypy.readthedocs.io/):

    mypy --strict .

### Updating setup

To install new packages, add them to `environment.yaml` then run:

    micromamba install -f environment.yaml

To add new pre-commit hooks, configure them in `.pre-commit-config.yaml` then run:

    pre-commit run --all-files
