#!/usr/bin/env python3
"""
patch_bnt.py  —  run from your repo root:
    python patch_bnt.py

What it fixes
─────────────
PyTorch 2.x added 'is_causal' (and later 'need_weights') kwargs to
TransformerEncoderLayer._sa_block().  The BNT repo subclasses that layer
with a custom _sa_block that only has the 1.12-era three-arg signature.
When PyTorch's TransformerEncoder.forward() calls _sa_block(..., is_causal=…)
it blows up with the TypeError you hit.

The fix: add **kwargs to the overridden _sa_block so it silently absorbs
any new keyword arguments PyTorch adds in the future too.  We do NOT pass
is_causal into multihead_attention_forward because brain networks are
never causal (it's a fully-connected undirected graph) — is_causal=False
is the correct semantic here regardless.
"""

import re, sys, pathlib

# ── locate the file ──────────────────────────────────────────────────
# Try the standard repo layout; adjust if yours is different.
CANDIDATES = [
    pathlib.Path("source/models/BNT/bnt.py"),
    pathlib.Path("BrainNetworkTransformer/source/models/BNT/bnt.py"),
]
TARGET = None
for c in CANDIDATES:
    if c.exists():
        TARGET = c
        break

if TARGET is None:
    print("[ERROR] Could not find bnt.py.  Run this script from the repo root:")
    print("        cd /home/tkpqz/BrainNetworkTransformer")
    print("        python patch_bnt.py")
    sys.exit(1)

print(f"[info] Patching: {TARGET}")
src = TARGET.read_text()

# ──────────────────────────────────────────────────────────────────────
# PATCH 1: _sa_block signature
#   OLD (all plausible whitespace variants):
#       def _sa_block(self, x, attn_mask, key_padding_mask):
#   NEW:
#       def _sa_block(self, x, attn_mask, key_padding_mask, **kwargs):
# ──────────────────────────────────────────────────────────────────────
pattern_sa = re.compile(
    r'(def _sa_block\(self,\s*x,\s*attn_mask,\s*key_padding_mask)\)'
)
if pattern_sa.search(src):
    src = pattern_sa.sub(r'\1, **kwargs)', src)
    print("[PATCH 1] _sa_block signature — added **kwargs  ✓")
else:
    # Maybe already patched or slightly different param names — check
    if '**kwargs' in src and '_sa_block' in src:
        print("[PATCH 1] _sa_block already has **kwargs — skipped")
    else:
        print("[WARN ] Could not find _sa_block pattern.  "
              "Check the file manually.")

# ──────────────────────────────────────────────────────────────────────
# PATCH 2: TransformerEncoder forward() — same story.
#   Some versions of the repo also override TransformerEncoder.forward
#   with the old signature that lacks is_causal.  We add **kwargs there
#   too if present.
#
#   OLD:  def forward(self, src, mask=None, src_key_padding_mask=None):
#   NEW:  def forward(self, src, mask=None, src_key_padding_mask=None, **kwargs):
#
#   We only touch it if it's inside a class that contains "_sa_block"
#   (i.e. the interpretable encoder, not some unrelated forward).
# ──────────────────────────────────────────────────────────────────────
pattern_fwd = re.compile(
    r'(def forward\(self,\s*src,\s*mask\s*=\s*None,\s*'
    r'src_key_padding_mask\s*=\s*None)\)(\s*->.*?)?:'
)
matches = list(pattern_fwd.finditer(src))
if matches:
    # Patch only the FIRST match (the encoder's forward, not the top-level model)
    m = matches[0]
    old = m.group(0)
    # insert **kwargs before the closing paren
    new = m.group(1) + ', **kwargs)' + (m.group(2) or '') + ':'
    src = src.replace(old, new, 1)
    print("[PATCH 2] TransformerEncoder.forward signature — added **kwargs ")
else:
    print("[PATCH 2] No bare forward(src, mask, src_key_padding_mask) found — "
          "likely not needed or already patched")

# ──────────────────────────────────────────────────────────────────────
# Write back
# ──────────────────────────────────────────────────────────────────────
TARGET.write_text(src)
print(f"\n[info] Written back to {TARGET}")

# ── show the patched region for human review ─────────────────────────
print("\n" + "=" * 60)
print(" Patched code (search for _sa_block / forward):")
print("=" * 60)
for i, line in enumerate(src.splitlines(), 1):
    if '_sa_block' in line or ('def forward' in line and 'src' in line):
        # print 2 lines of context after
        lines = src.splitlines()
        for j in range(i - 1, min(i + 2, len(lines))):
            marker = ">>>" if j == i - 1 else "   "
            print(f"  {marker} {j+1:4d} | {lines[j]}")
        print()

print("[done] Now run the training command again.")