#!/usr/bin/env bash
# Download the pinned OPA release binary into tools/opa (git-ignored) and verify its sha256.
# Idempotent: does nothing when tools/opa already has the pinned hash.
# The Rego rendering's engine check (src/authinv/equivalence/rego.py) runs this binary.
set -euo pipefail

OPA_VERSION="v1.21.1"

# bash 3.2 (macOS) has no associative arrays, hence the case table.
case "$(uname -s)_$(uname -m)" in
  Darwin_arm64)
    asset="darwin_arm64_static"
    want="a00a6469a0968c47c01137ed0dacaa88148be5d0eaa8524310cd371f3a98a169" ;;
  Linux_x86_64)
    asset="linux_amd64_static"
    want="668506eb17a2eaa1fce6cc0d1f42ef85125d4ac5bda5fc74d1152d0c77145031" ;;
  *) echo "fetch_opa: no pinned OPA build for $(uname -s)_$(uname -m)" >&2; exit 1 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/tools/opa"
mkdir -p "$root/tools"

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

if [ -x "$dest" ] && [ "$(sha "$dest")" = "$want" ]; then
  echo "fetch_opa: $dest is OPA $OPA_VERSION ($asset), sha256 ok"
  exit 0
fi

tmp="$(mktemp "$root/tools/opa.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
url="https://github.com/open-policy-agent/opa/releases/download/$OPA_VERSION/opa_$asset"
echo "fetch_opa: downloading $url"
curl -fsSL --retry 3 -o "$tmp" "$url"
got="$(sha "$tmp")"
if [ "$got" != "$want" ]; then
  echo "fetch_opa: sha256 mismatch for opa_$asset: got $got, want $want" >&2
  exit 1
fi
chmod +x "$tmp"
mv "$tmp" "$dest"
trap - EXIT
echo "fetch_opa: installed OPA $OPA_VERSION ($asset) at $dest"
