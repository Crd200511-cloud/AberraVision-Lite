"""Execute the lesson with the project kernel and preserve its outputs."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for variable, directory in {
    "IPYTHONDIR": "ipython", "JUPYTER_RUNTIME_DIR": "jupyter",
    "MPLCONFIGDIR": "matplotlib", "XDG_CACHE_HOME": "general",
}.items():
    path = ROOT / ".cache" / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)

import nbformat
from nbclient import NotebookClient

path = ROOT / "01_optics_learning.ipynb"
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=90, kernel_name="aberravision",
               resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, path)
print(f"Executed {sum(c.cell_type == 'code' for c in nb.cells)} code cells: {path.name}")
