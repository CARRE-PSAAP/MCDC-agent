import ast, json
import sys
from pathlib import Path
from typing import Dict, List, Any

# CONFIGURATION
# REPO_ROOT is virtual now
OUTPUT_JSON = Path(__file__).parent / "scraped_docs/auto_api.json"

# Import GitHub helpers
sys.path.append(str(Path(__file__).parent))
try:
    from scrape_api import fetch_github_file_tree, fetch_raw_content
except ImportError:
    # Fallback/Mock if running standalone without context
    print("Warning: Could not import GitHub helpers.")
    fetch_github_file_tree = lambda x: []
    fetch_raw_content = lambda x: ""


def format_sig(node: ast.FunctionDef) -> str:
    """Reconstruct readable signature from AST."""
    args = []
    # positional args
    for arg in node.args.args:
        if arg.annotation:
            try:
                args.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
            except:
                args.append(arg.arg)
        else:
            args.append(arg.arg)
    # defaults (applied from right)
    if node.args.defaults and args:
        for i, d in enumerate(node.args.defaults):
            idx = len(args) - len(node.args.defaults) + i
            if 0 <= idx < len(args):
                try:
                    args[idx] += f" = {ast.unparse(d)}"
                except:
                    pass
    # *args
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    # **kwargs
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return f"def {node.name}({', '.join(args)})"


def extract_from_node(node: ast.AST, pkg_name: str) -> Dict[str, Any]:
    """Get name, docstring, params from function or class."""
    data = {"name": node.name, "docstring": ast.get_docstring(node) or ""}

    if isinstance(node, ast.FunctionDef):
        data["type"] = "function"
        data["signature"] = format_sig(node)
        data["parameters"] = [
            {"name": arg.arg, "has_default": arg in node.args.defaults}
            for arg in node.args.args
        ]

    elif isinstance(node, ast.ClassDef):
        data["type"] = "class"
        # look at __init__ for constructor params
        init = next(
            (
                m
                for m in node.body
                if isinstance(m, ast.FunctionDef) and m.name == "__init__"
            ),
            None,
        )
        if init:
            data["init_signature"] = format_sig(init)
            data["init_parameters"] = [
                {"name": arg.arg, "has_default": arg in init.args.defaults}
                for arg in init.args.args[1:]  # skip 'self'
            ]
        else:
            data["init_signature"] = ""
            data["init_parameters"] = []
    return data


def walk_package_github(pkg_prefix: str = "mcdc/") -> Dict[str, Any]:
    """Walk every .py file in the package via GitHub API; extract all public functions/classes."""
    api: Dict[str, Any] = {}

    print(f"Fetching file tree for {pkg_prefix}...")
    files = fetch_github_file_tree(pkg_prefix)
    
    # Filter for .py files
    py_files = [f for f in files if f["path"].endswith(".py")]
    print(f"Found {len(py_files)} Python files.")

    for file_info in py_files:
        path_str = file_info["path"]
        
        # Skip private modules and test files
        parts = path_str.split("/")
        if any(part.startswith("_") for part in parts) or "test" in parts:
            continue
        
        # Skip internal utility packages
        if any(p in parts for p in ["mcdc_get", "mcdc_set", "code_factory", "object_", "type_"]):
            continue

        # Construct module name: mcdc/main.py -> mcdc.main
        module_name = path_str.replace("/", ".").replace(".py", "")
        
        try:
            content = fetch_raw_content(path_str)
            if not content:
                continue
            tree = ast.parse(content)
        except SyntaxError:
            print(f"⚠  Skip {path_str} – parse error")
            continue
        except Exception as e:
            print(f"⚠  Skip {path_str} – {e}")
            continue

        # Module-level functions and classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                key = f"{module_name}.{node.name}"
                api[key] = extract_from_node(node, key)

    return api


def main():
    print(f"=== Generating API Docs from GitHub Source ===")

    api = walk_package_github("mcdc/")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(api, indent=2), encoding="utf-8")
    print(f"✓ Extracted {len(api)} APIs → {OUTPUT_JSON}")

    # Show what's missing from RTD
    rtd_funcs = {
        "material",
        "nuclide",
        "cell",
        "lattice",
        "surface",
        "universe",
        "eigenmode",
        "setting",
        "source",
        "tally",
    }
    # Auto funcs: look for top-level mcdc.X (e.g. mcdc.material)
    # The scraped module names will likely be mcdc.type_.material.material_card...
    # We want to see if we captured the core concepts.
    
    # Simple check for now
    print(f"Stats: {len(api)} total API entities found.")


if __name__ == "__main__":
    main()
