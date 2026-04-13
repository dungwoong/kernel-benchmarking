# BASH_SOURCE[0] points to path of add_path.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/python:$REPO_ROOT/third_party"