#!/usr/bin/env bash
if [[ "$(basename -- "$0")" == "setup.sh" ]]; then
    >&2 echo "Don't run $0, source it"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ $(which python3) == "${SCRIPT_DIR}/.env/bin/python3" ]]; then
    echo "already in virtual env"
    return
fi

if [[ ! -d ".env/" ]]; then
    echo "installing python environment"
    python3.10 -m venv .env
    source .env/bin/activate 
    pip3 install -r requirements.txt
else
    echo "activating python environment"
    source .env/bin/activate 
fi

echo "setting python path to $SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"