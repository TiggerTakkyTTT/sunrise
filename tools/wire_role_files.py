#!/usr/bin/env python3
"""One-off: write one role file per trait of the old IMPORTS/EXPORTS lists of
crates/sunrise-mod-api-gen/src/wire/mod.rs into crates/sunrise-mod-vocab/src/wire/traits/,
and turn the discovered `.export` files whose trait the game also hands back to a mod into `.held`.

Each file's text is one line: the first sentence of the trait's doc comment when the trait is
written in the tree, else a plain sentence saying what crosses. Run from the repository root;
`git rm` this script and the lists in the hand-fix commits that follow.
"""
import re
import subprocess
from pathlib import Path

GEN = Path("crates/sunrise-mod-api-gen/src/wire/mod.rs")
TRAITS = Path("crates/sunrise-mod-vocab/src/wire/traits")


def listed(source, name):
    body = re.search(r"const " + name + r": &\[&str\] = &\[(.*?)\n\];", source, re.S).group(1)
    return re.findall(r'"([^"]+)"', body)


def doc_of(name):
    hits = subprocess.run(
        ["rg", "-n", "--no-heading", rf"^\s*pub trait {name}\b", "crates/sunrise-mod-vocab/src", "crates/sunrise-mod-api/src", "-g", "*.rs",
         ],
        capture_output=True, text=True).stdout.splitlines()
    if not hits:
        return None
    path, line = hits[0].split(":")[:2]
    lines = Path(path).read_text().splitlines()[: int(line) - 1]
    doc = []
    for text in reversed(lines):
        text = text.strip()
        if text.startswith("#["):
            continue
        if not text.startswith("///"):
            break
        doc.append(text[3:].strip())
    doc.reverse()
    paragraph = []
    for text in doc:
        if not text:
            break
        paragraph.append(text)
    first = re.split(r"(?<=\.)\s", " ".join(paragraph), maxsplit=1)[0].strip().rstrip(":")
    if not first:
        return None
    return first if first.endswith(".") else first + "."


def main():
    source = GEN.read_text()
    for role, names in (("import", listed(source, "IMPORTS")), ("export", listed(source, "EXPORTS"))):
        for name in names:
            target = TRAITS / f"{name}.{role}"
            assert not target.exists(), target
            if role == "import":
                fallback = f"A trait of the game's that a mod calls: the methods of `{name}` cross as imports."
            else:
                fallback = f"A trait a mod implements: the methods of `{name}` cross as exports."
            target.write_text((doc_of(name) or fallback) + "\n")
    for old in sorted(TRAITS.glob("*.export")):
        if old.name in HELD:
            subprocess.run(["git", "mv", str(old), str(old.with_suffix(".held"))], check=True)


# Discovered exports the game also hands back to a mod (the `held-<interface>` imports).
HELD = {
    "CraftingRecipeHandler.export", "CrashSection.export", "EntityRenderLayer.export",
    "EntityRenderer.export", "GlobalLootModifier.export", "GlobalLootModifierSerializer.export",
    "IPayloadHandler.export", "ItemContainerProvider.export", "PayloadWork.export",
    "SpecialModelRenderer.export", "StartupScreen.export", "SuggestionProvider.export",
}

if __name__ == "__main__":
    main()
