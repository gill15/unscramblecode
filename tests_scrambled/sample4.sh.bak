#!/usr/bin/env bash
set -euo pipefail

copy_file(){
local src="$1"
local dst="$2"
mkdir -p "$(dirname "$dst")"
cp -a "$src" "$dst"
}

main(){
if [[ $# -ne 2 ]]; then
echo "usage: $0 <src> <dst>" >&2
exit 2
fi
copy_file "$1" "$2"
}

main "$@"
