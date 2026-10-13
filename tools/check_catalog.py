#!/usr/bin/env python3
"""Cross-check the three places that describe an effect parameter's slot:

  1. the `// @fx kernel: name@slot[:width] ...` annotation above each kernel in Sources/Effects/FX_*.metal
  2. the PARAM()/PARAM2/3/4() reads inside that kernel's body
  3. the parameter list of every entry in Sources/Effects/EffectCatalog.swift (slot = sum of earlier widths)

Errors it catches: overlapping slots (e.g. a colour at @0:4 followed by a scalar at @1), a kernel reading a slot its
annotation does not declare, an annotated slot nobody reads, a catalog parameter whose name/slot/width disagrees with
the kernel, a parameter no kernel reads that is not marked as a runner parameter, defaults outside min...max.

usage: check_catalog.py          (exit status 1 if anything is wrong)
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FX = ROOT / "Sources" / "Effects"
CATALOG = FX / "EffectCatalog.swift"
EXPECTED = {"Blur & Sharpen": 7, "Color Correction": 7, "Distort": 7, "Generate": 7,
            "Simulation": 6, "Stylize": 7, "Perspective": 5, "Keying": 5}
CATEGORY = {"blurSharpen": "Blur & Sharpen", "colorCorrection": "Color Correction", "distort": "Distort",
            "generate": "Generate", "simulation": "Simulation", "stylize": "Stylize",
            "perspective": "Perspective", "keying": "Keying"}
WIDTH = {"P": 1, "P2": 2, "P4": 4, "PM3": 9}

errors, warnings = [], []
def err(msg): errors.append(msg)
def warn(msg): warnings.append(msg)

# ----------------------------------------------------------------------------- small parsing helpers
def match_close(text, i, open_ch, close_ch):
    """Index of the bracket closing the one at text[i], skipping string literals."""
    depth, j, in_str = 0, i, False
    while j < len(text):
        c = text[j]
        if in_str:
            if c == "\\": j += 1
            elif c == '"': in_str = False
        elif c == '"': in_str = True
        elif c == open_ch: depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0: return j
        j += 1
    return -1

def split_args(s):
    out, depth, cur, in_str = [], 0, "", False
    for c in s:
        if c == '"': in_str = not in_str
        if not in_str:
            if c in "([": depth += 1
            elif c in ")]": depth -= 1
            elif c == "," and depth == 0:
                out.append(cur.strip()); cur = ""; continue
        cur += c
    if cur.strip(): out.append(cur.strip())
    return out

def num(s):
    try: return float(s)
    except (TypeError, ValueError): return None

# ----------------------------------------------------------------------------- 1+2: the Metal side
ANNOT = re.compile(r"^//\s*@fx\s+(\w+):(.*)$", re.M)
TOKEN = re.compile(r"(\w+)@(\d+)(?::(\d+))?")
READ = re.compile(r"\bPARAM([234]?)\(\s*(\d+)\s*\)")

def kernel_body(src, name):
    m = re.search(r"\b(?:kernel|vertex|fragment)\s+[\w:<>]+\s+" + re.escape(name) + r"\s*\(", src)
    if not m: return None
    close = match_close(src, m.end() - 1, "(", ")")
    brace = src.find("{", close)
    end = match_close(src, brace, "{", "}")
    return src[brace:end + 1]

kernels = {}      # name -> {"params": {name: (slot, width)}, "file": str}
all_kernel_names = set()
for f in sorted(FX.glob("FX_*.metal")):
    src = f.read_text()
    all_kernel_names |= set(re.findall(r"\b(?:kernel|vertex|fragment)\s+[\w:<>]+\s+(\w+)\s*\(", src))
    for m in ANNOT.finditer(src):
        kname, rest = m.group(1), m.group(2)
        params = {}
        for t in TOKEN.finditer(rest):
            pname, slot, width = t.group(1), int(t.group(2)), int(t.group(3) or 1)
            if pname in params: err(f"{f.name}: {kname}: parameter '{pname}' annotated twice")
            params[pname] = (slot, width)
        kernels[kname] = {"params": params, "file": f.name}

        # overlapping slots inside one annotation
        owner = {}
        for pname, (slot, width) in params.items():
            for s in range(slot, slot + width):
                if s in owner:
                    err(f"{f.name}: {kname}: slot {s} is claimed by both '{owner[s]}' and '{pname}'")
                owner[s] = pname

        body = kernel_body(src, kname)
        if body is None:
            err(f"{f.name}: annotation for '{kname}' but no such kernel"); continue
        read = set()
        for r in READ.finditer(body):
            w = int(r.group(1) or 1); s = int(r.group(2))
            read |= set(range(s, s + w))
        for s in sorted(read - set(owner)):
            err(f"{f.name}: {kname} reads PARAM slot {s} which its annotation does not declare")
        for s in sorted(set(owner) - read):
            err(f"{f.name}: {kname}: '{owner[s]}' (slot {s}) is annotated but never read")

# ----------------------------------------------------------------------------- 3: the Swift catalog
cat_src = CATALOG.read_text()

def find_param_block(name):
    """Text of `func name(...) { ... }` or `let name = [...]` anywhere in the catalog file."""
    m = re.search(r"func\s+" + re.escape(name) + r"\s*\(", cat_src)
    if m:
        close = match_close(cat_src, m.end() - 1, "(", ")")
        brace = cat_src.find("{", close)
        return cat_src[brace:match_close(cat_src, brace, "{", "}") + 1]
    m = re.search(r"let\s+" + re.escape(name) + r"\b[^=]*=\s*\[", cat_src)
    if m:
        i = m.end() - 1
        return cat_src[i:match_close(cat_src, i, "[", "]") + 1]
    return None

entries = []
for m in re.finditer(r"\bFX\(\s*\"(\w+)\"\s*,\s*\"([^\"]+)\"\s*,\s*\.(\w+)", cat_src):
    start = cat_src.index("(", m.start())
    text = cat_src[m.start():match_close(cat_src, start, "(", ")") + 1]
    e = {"id": m.group(1), "name": m.group(2), "category": CATEGORY.get(m.group(3), m.group(3)), "kernels": [],
         "params": [], "runner": []}
    e["kernels"] += re.findall(r"\.pass\(\s*\"(\w+)\"", text)
    p = re.search(r"particles:\s*\(\s*\"(\w+)\"\s*,\s*\"(\w+)\"\s*\)", text)
    if p: e["kernels"] += [p.group(1), p.group(2)]
    r = re.search(r"runner:\s*\[([^\]]*)\]", text)
    if r: e["runner"] = re.findall(r"\"(\w+)\"", r.group(1))

    pm = re.search(r"\bparams:\s*", text)
    if not pm:
        err(f"catalog: {e['id']} has no params:"); entries.append(e); continue
    after = text[pm.end():]
    if after.startswith("["):
        block = after[:match_close(after, 0, "[", "]") + 1]
    else:
        call = re.match(r"(\w+)\(", after)
        block = find_param_block(call.group(1)) if call else None
        if block is None:
            err(f"catalog: {e['id']}: cannot resolve params list"); entries.append(e); continue
    slot = 0
    for pmatch in re.finditer(r"\b(PM3|P2|P4|P)\(\s*\"(\w+)\"", block):
        fn, pname = pmatch.group(1), pmatch.group(2)
        i = block.index("(", pmatch.start())
        args = split_args(block[i + 1:match_close(block, i, "(", ")")])
        width = WIDTH[fn]
        e["params"].append({"name": pname, "slot": slot, "width": width, "fn": fn, "args": args})
        slot += width
    entries.append(e)

ids = [e["id"] for e in entries]
for dup in {i for i in ids if ids.count(i) > 1}: err(f"catalog: duplicate effect id '{dup}'")

used_kernels = set()
for e in entries:
    names = [p["name"] for p in e["params"]]
    for dup in {n for n in names if names.count(n) > 1}: err(f"catalog: {e['id']}: duplicate parameter '{dup}'")
    by_name = {p["name"]: p for p in e["params"]}
    covered = set()
    for kname in e["kernels"]:
        used_kernels.add(kname)
        if kname not in all_kernel_names:
            err(f"catalog: {e['id']}: kernel '{kname}' does not exist in FX_*.metal"); continue
        k = kernels.get(kname)
        if k is None:
            err(f"catalog: {e['id']}: kernel '{kname}' has no '// @fx' annotation"); continue
        for pname, (slot, width) in k["params"].items():
            p = by_name.get(pname)
            if p is None:
                err(f"catalog: {e['id']}: kernel {kname} reads '{pname}'@{slot} but the entry has no such parameter")
            elif (p["slot"], p["width"]) != (slot, width):
                err(f"catalog: {e['id']}: '{pname}' is @{p['slot']}:{p['width']} in the catalog but @{slot}:{width} in {kname}")
            else:
                covered.add(pname)
    for rn in e["runner"]:
        if rn not in by_name: err(f"catalog: {e['id']}: runner parameter '{rn}' does not exist")
    for p in e["params"]:
        if p["name"] not in covered and p["name"] not in e["runner"]:
            err(f"catalog: {e['id']}: parameter '{p['name']}' (slot {p['slot']}) is read by no kernel and is not listed in runner:")
        if p["name"] in covered and p["name"] in e["runner"]:
            warn(f"catalog: {e['id']}: '{p['name']}' is listed in runner: but a kernel also reads it")
        # defaults / ranges (literals only)
        if p["fn"] == "P":
            a = p["args"]
            d, lo, hi = (num(a[1]) if len(a) > 1 else None), (num(a[2]) if len(a) > 2 else None), (num(a[3]) if len(a) > 3 else None)
            if None not in (d, lo, hi) and not (lo <= d <= hi):
                err(f"catalog: {e['id']}.{p['name']}: default {d} outside {lo}...{hi}")
            if lo is not None and hi is not None and lo >= hi:
                err(f"catalog: {e['id']}.{p['name']}: min {lo} >= max {hi}")
            kind = a[4] if len(a) > 4 else ".scalar"
            if kind == ".choice":
                if len(a) < 6: err(f"catalog: {e['id']}.{p['name']}: .choice without options")
                elif a[5].startswith("[") and None not in (hi,):
                    n = len(split_args(a[5][1:-1]))
                    if hi != n - 1: err(f"catalog: {e['id']}.{p['name']}: max {hi} but {n} options")

for kname, k in kernels.items():
    if kname not in used_kernels:
        warn(f"{k['file']}: annotated kernel '{kname}' is not used by any catalog entry")

counts = {}
for e in entries: counts[e["category"]] = counts.get(e["category"], 0) + 1
for cat, n in EXPECTED.items():
    if counts.get(cat, 0) != n: err(f"catalog: category '{cat}' has {counts.get(cat, 0)} effects, expected {n}")
if len(entries) != sum(EXPECTED.values()):
    err(f"catalog: {len(entries)} effects, expected {sum(EXPECTED.values())}")

# ----------------------------------------------------------------------------- report
for w in warnings: print("warn:", w)
for e in errors: print("ERROR:", e)
slots = sum(len(e["params"]) for e in entries)
print(f"\n{len(entries)} effects, {slots} parameters, {len(kernels)} annotated kernels, "
      f"{len(errors)} errors, {len(warnings)} warnings")
sys.exit(1 if errors else 0)
