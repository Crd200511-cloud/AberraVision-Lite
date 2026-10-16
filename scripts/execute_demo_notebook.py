"""Execute the saved-model Notebook without running training again."""
import argparse
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for variable,directory in {'IPYTHONDIR':'ipython','JUPYTER_RUNTIME_DIR':'jupyter',
                           'MPLCONFIGDIR':'matplotlib','XDG_CACHE_HOME':'general'}.items():
    path=ROOT/'.cache'/directory; path.mkdir(parents=True,exist_ok=True)
    os.environ[variable]=str(path)

import nbformat
from nbclient import NotebookClient

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kernel',default='python3')
args=parser.parse_args()
path=ROOT/'demo.ipynb'
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=120,kernel_name=args.kernel,resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.validate(nb)
nbformat.write(nb,path)
print(f'Executed {sum(c.cell_type=="code" for c in nb.cells)} cells successfully.')
