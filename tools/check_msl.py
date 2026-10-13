#!/usr/bin/env python3
"""Syntax/type-check B.P's .metal sources with clang++ and a stub <metal_stdlib>.

There is no Metal compiler outside Xcode, so this copies the shader folders to a scratch directory,
rewrites MSL vector constructors (`float4(...)` -> `__mk<float4>(...)`, which plain C++ lacks) and runs
`clang++ -fsyntax-only` against tools/mslshim. It catches typos, undeclared names, wrong argument
types/counts and vector-width mismatches. It cannot catch Metal-only rules (address-space errors,
threadgroup limits, unsupported pixel formats) - the Xcode build remains the final judge.

usage: check_msl.py [file.metal ...]      (default: every .metal under Sources/)
"""
import pathlib, re, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHIM = ROOT / "tools" / "mslshim"
CTOR = re.compile(r"\b(float|int|uint|half|short|ushort|char|uchar|bool)([234])(?!x)\s*\(")

def rewrite(text: str) -> str:
    return CTOR.sub(lambda m: f"__mk<{m.group(1)}{m.group(2)}>(", text)

def main() -> int:
    src = ROOT / "Sources"
    targets = [pathlib.Path(a).resolve() for a in sys.argv[1:]] or sorted(src.rglob("*.metal"))
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp) / "Sources"
        for f in list(src.rglob("*.metal")) + list(src.rglob("*.h")):
            dest = work / f.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(rewrite(f.read_text()))
        incs = sorted({str(p.parent) for p in work.rglob("*.h")})
        failed = 0
        for t in targets:
            rel = work / t.relative_to(src)
            cmd = ["clang++", "-std=c++17", "-x", "c++", "-fsyntax-only", "-ferror-limit=40",
                   "-Wno-unknown-attributes", "-Wno-unused-value", "-Wno-ignored-attributes",
                   "-I", str(SHIM), *sum((["-I", i] for i in incs), []), str(rel)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            out = (r.stderr or "").replace(str(work), "Sources")
            errs = [l for l in out.splitlines() if " error:" in l]
            print(f"{'FAIL' if r.returncode else 'ok  '}  {t.relative_to(src)}"
                  + (f"   ({len(errs)} errors)" if errs else ""))
            if r.returncode:
                failed += 1
                shown = set()
                lines = out.splitlines()
                for i, l in enumerate(lines):
                    if " error:" in l and l not in shown:
                        shown.add(l)
                        print("   " + l)
                        if i + 1 < len(lines) and not lines[i + 1].startswith(("In file", "Sources", "/")):
                            print("      " + lines[i + 1].strip())
        return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
