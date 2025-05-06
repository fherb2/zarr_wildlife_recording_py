#!/bin/bash
set -e  # Stop on error

echo "Check if there is a virtula Python environment active..."
# Prüfe auf aktives conda- oder venv-Environment
if (command -v conda >/dev/null && [ -n "$CONDA_DEFAULT_ENV" ]) || [ -n "$VIRTUAL_ENV" ]; then
    echo
    echo "Sorry: A virtual Python environment is active:"
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "  → Conda-Umgebung: $CONDA_DEFAULT_ENV"
    fi
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "  → venv/virtualenv: $VIRTUAL_ENV"
    fi
    echo
    echo "Please, deactivate it first (eg. 'conda deactivate' oder 'deactivate')"
    echo "and restart this script after this."
    exit 1
fi
echo "No activated Python environment. We can progress..."

echo "Import Python- und Poetry-Version from pyproject.toml ..."
REQUIRED_PYTHON_VERSION=$(grep -Po 'python\s*=\s*"\^?\K[0-9]+\.[0-9]+' pyproject.toml | head -1)
REQUIRED_POETRY_VERSION=$(grep -Po 'poetry-version\s*=\s*"\K[0-9]+\.[0-9]+\.[0-9]+' pyproject.toml || true)
echo Done.
echo
echo "Ok. The 'pyproject.toml' says:"
echo "    Needed Python version: $REQUIRED_PYTHON_VERSION"
[[ -n "$REQUIRED_POETRY_VERSION" ]] && echo "    Needed Poetry version: $REQUIRED_POETRY_VERSION"
echo "See, if we found the right installations and versions during following checks."
echo
echo Check pyenv...
if ! command -v pyenv &> /dev/null; then
    echo "pyenv not found. Please install: https://github.com/pyenv/pyenv"
    exit 1
fi
echo Done.
echo
echo Check Poetry...
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install version $REQUIRED_POETRY_VERSION from: https://python-poetry.org/docs/#installation"
    exit 1
fi
echo Done.
echo
echo Check Poetry version...
if [[ -n "$REQUIRED_POETRY_VERSION" ]]; then
    POETRY_VERSION=$(poetry --version | awk '{print $3}')
    if [[ "$POETRY_VERSION" != "$REQUIRED_POETRY_VERSION" ]]; then
        echo "Poetry-Version is $POETRY_VERSION, expected is $REQUIRED_POETRY_VERSION"
    fi
fi
echo Done.
echo
echo "Check if required Python version is installed via pyenv..."
if ! pyenv versions --bare | grep -q "^$REQUIRED_PYTHON_VERSION$"; then
    echo "Missing Python version $REQUIRED_PYTHON_VERSION. Install it via pyenv now..."
    pyenv install "$REQUIRED_PYTHON_VERSION"
fi
echo Ok. Set Python version locally for this project...
pyenv local "$REQUIRED_PYTHON_VERSION"
echo Done.
echo
echo "Configure Poetry to set the Python environment inside this project (directory '.venv')..."
poetry config virtualenvs.in-project true
poetry env use "$REQUIRED_PYTHON_VERSION"
poetry install
echo Done.
echo
echo Ok. Ready!
echo 
