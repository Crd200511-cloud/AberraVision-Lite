"""Create a local source/artifact archive; exclude third-party photos and caches."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=set()
    for pattern in ['*.py','*.md','requirements*.txt','config.json','.gitignore','*.ipynb',
                    'src/*.py','scripts/*.py','tests/*.py','docs/*.md',
                    'models/*.pt','models/*.json','models/*.md']:
        files.update(ROOT.glob(pattern))
    for folder in ['validation','data_preparation','formal/run_01','test/run_01',
                   'demo','cnn_pilot/run_01','learning','wiener_learning']:
        for file in (ROOT/'results'/folder).glob('*'):
            if file.is_file() and file.suffix in {'.json','.csv','.png'} and file.name!='preview_data.json':
                files.add(file)
    files.update((ROOT/'results/cnn_pilot/run_01/source_snapshot').rglob('*.py'))
    names=[str(p.relative_to(ROOT)) for p in files]
    if any(n.startswith(('data/','results/data_preview/','results/test_arrays/','.venv/','.cache/')) for n in names):
        raise ValueError('Excluded data accidentally entered package.')
    dist=ROOT/'dist'; dist.mkdir(exist_ok=True)
    target=dist/'AberraVision-Lite.zip'
    manifest=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            data=path.read_bytes(); name=str(path.relative_to(ROOT))
            # Avoid exposing absolute local workspace paths in historical notebook outputs.
            if path.suffix=='.ipynb':
                data=data.decode().replace(str(ROOT),'.').encode()
            archive.writestr('AberraVision-Lite/'+name,data)
            manifest.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    (dist/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{target}: {len(files)} files, {target.stat().st_size/1e6:.2f} MB')


if __name__=='__main__': main()
