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

(In case micromamba fails to install pip packages, you can install them manually
by running `micromamba activate nerd` then `pip install --editable .[dev]`)

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

### Building documentation

The documentation site is developed in the `./docs` directory using [Sphinx](https://www.sphinx-doc.org/en/master/usage/index.html).

To build the docs:

```bash
cd docs
# generate the API docs (this pulls information from the code and docstrings)
sphinx-apidoc -M -o source/api ../src/nird/ --force
# build the documentation site
make html
```

To preview the site:

```
cd build/html
python -m http.server
```

Then open a browser at the address shown (e.g. `http://0.0.0.0:8000`).

## Docker and DAFNI

```bash
# Build the image - run this from the root of the repository
docker build -f ./containers/nird_road/Dockerfile -t nismod/nird_road:latest .
```

```bash
# Run image using test data - run this from the data directory
docker run --rm -v ${PWD}/test_inputs:/data/inputs -v ${PWD}/test_outputs:/data/outputs --env NUMBER_CPUS=1 --env DEPTH_THRESHOLD=30 nismod/nird_road
```

```bash
# Save image to file
docker save -o nird_road.tar nismod/nird_road:latest
```

## Acknowledgments

This project was developed as part of a UKRI-funded research grant, reference ST/Y003780/1,
supported by STFC and DAFNI, within the "Building a Secure and Resilient World" theme.
