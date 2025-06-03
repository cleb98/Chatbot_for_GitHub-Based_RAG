import os, shutil, ast, yaml
from pathlib import Path
import git
from config import Configurator
import re
from pathlib import Path
from typing import List, Tuple



# === CLONE REPO ===
def clone_repo(repo_url: str, target_dir: str):
    shutil.rmtree(target_dir, ignore_errors=True)
    git.Repo.clone_from(repo_url, target_dir)
    print(f"✅ Clonata {repo_url}")


# === PARSE .md ===
# === REGEX PER MARKDOWN: define i titoli e i separatori da usare per dividere i blocchi ===
TITLE_RE = re.compile(r'^\s*(#{1,6})\s+(.*)')       #  #, ##, ### …
HR_RE    = re.compile(r'^\s*(\*{3,}|-{3,}|_{3,})\s*$')  # ***  ---  ___

def extract_markdown_blocks(md_path: Path, use_separators: bool = True) -> List[Tuple[str, str]]:
    """
    Divide un file .md in blocchi titolo+corpo.
    Se `use_separators` è True, considera anche linee *** / --- / ___ come separatori logici.
    Ritorna [(title, body), ...] dove `body` include il titolo.
    """
    blocks, current_lines, current_title = [], [], None

    with open(md_path, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            is_title = TITLE_RE.match(line)
            is_sep = HR_RE.match(line) if use_separators else False

            if is_title or is_sep:
                if current_lines:
                    blocks.append((
                        current_title or "untitled",
                        "".join(current_lines).strip()
                    ))
                    current_lines = []

                if is_title:
                    current_title = is_title.group(2).strip()
                elif is_sep:
                    current_title = "separator"

            current_lines.append(line)

        if current_lines:
            blocks.append((
                current_title or "untitled",
                "".join(current_lines).strip()
            ))

    return blocks


# === ESTRAZIONE BLOCCO PYTHON da file .py ===
def extract_python_chunks_linear(file_path: Path):
    """
    Estrae i blocchi di codice Python da un file, mantenendo le informazioni di riga.
    """
    with open(file_path, "r", encoding="utf8", errors="ignore") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    blocks = []

    seen_imports = set()

    # 1. IMPORTS (line-based, non AST)
    import_block_lines = []
    import_start_line = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import") or stripped.startswith("from"):
            if stripped not in seen_imports:
                if import_start_line is None:
                    import_start_line = i + 1
                import_block_lines.append(stripped)
                seen_imports.add(stripped)

    # Unico blocco per tutti gli import
    if import_block_lines:
        code_block = "```python\n" + "\n".join(import_block_lines) + "\n```"
        blocks.append({
            "body": code_block,
            "lineno": import_start_line
        })

    # 2. AST-level top-level blocks
    for node in ast.iter_child_nodes(tree):
        start = node.lineno - 1
        end = getattr(node, "end_lineno", start + 1)
        body = "\n".join(lines[start:end])
        code_block = f"```python\n{body.strip()}\n```"

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Assign, ast.AnnAssign, ast.If)):
            # Solo "__main__" è un caso speciale
            if isinstance(node, ast.If):
                try:
                    if not (
                        isinstance(node.test, ast.Compare) and
                        isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__" and
                        isinstance(node.test.ops[0], ast.Eq) and
                        isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == "__main__"
                    ):
                        continue
                except Exception:
                    continue
            blocks.append({
                "body": code_block,
                "lineno": node.lineno
            })

    # Ordina tutto per riga
    blocks.sort(key=lambda b: b["lineno"])
    return [b["body"] for b in blocks]




# === COSTRUISCE FILE .md AGGREGATO ===
def write_parsed_md(repo_dir: str, output_md: str, allowed_exts: set):
    """
    Aggrega i contenuti di tutti i file .py e .md in un unico file markdown strutturato.
    - Ogni file ha un'intestazione # <path>
    - I blocchi di codice Python sono formattati come ```python
    """
    out_path = Path(output_md)
    out_path.unlink(missing_ok=True)

    with open(out_path, "w", encoding="utf8") as out:
        for f in Path(repo_dir).rglob("*"):
            if f.suffix not in allowed_exts:
                continue

            relative_path = f.relative_to(repo_dir)
            out.write(f"# file:  {relative_path}\n\n")

            # Estrazione dei blocchi
            if f.suffix == ".py":
                try:
                    chunks = extract_python_chunks_linear(f)
                except Exception as e:
                    print(f"⚠️ Errore nel parsing di {f}: {e}")
                    continue

            elif f.suffix == ".md":
                try:
                    chunks = extract_markdown_blocks(f)
                except Exception as e:
                    print(f"⚠️ Errore nel parsing di {f}: {e}")
                    continue

            else:
                continue

            # Scrittura dei blocchi nel file aggregato
            for body in chunks:
                out.write(f"{body}\n\n")

            # Separatore tra file (opzionale)
            out.write("\n---\n\n")

    print(f"✅ File aggregato scritto in: {output_md}")

# === MAIN ===
def main():
    cfg = Configurator("config.yaml")
    #se esiste la cartella temporanea, non clona il repo
    if not Path(cfg.local_repo_dir).exists():
        clone_repo(cfg.repo_url, cfg.local_repo_dir)
    else:
        print(f"⚠️ Cartella {cfg.local_repo_dir} esistente, non clono il repo.")
    write_parsed_md(cfg.local_repo_dir, cfg.output_md_path, cfg.include_extensions)

if __name__ == "__main__":
    main()
