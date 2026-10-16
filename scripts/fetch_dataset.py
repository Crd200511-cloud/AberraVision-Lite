"""Acquire the 150 preselected source photos, with resumable checksum checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_div2k_preview import RemoteZip, URL


def main():
    directory = ROOT/'results/data_preparation'
    split_bytes = (directory/'candidate_split.json').read_bytes()
    manifest = json.loads(split_bytes)
    selected = [(group, image_id) for group, ids in manifest['splits'].items() for image_id in ids]
    ids = [i for _, i in selected]
    if len(ids) != 150 or len(set(ids)) != 150 or any(not i.isdigit() or len(i) != 4 for i in ids):
        raise ValueError('Expected 150 distinct four-digit source IDs.')
    record_path = directory/'dataset_acquisition.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    known = {r['id']: r for r in previous.get('files', [])}
    preview = json.loads((directory/'preview_acquisition.json').read_text())
    known.update({r['id']: r for r in preview['files']})
    output = ROOT/'data/raw/DIV2K_train_HR'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with RemoteZip(URL) as remote, zipfile.ZipFile(remote) as archive:
        expected_etag = previous.get('archive_etag', preview['archive_etag'])
        if remote.etag != expected_etag:
            raise RuntimeError('Official archive changed since preview acquisition; inspect before continuing.')
        for index, (group, image_id) in enumerate(selected, start=1):
            member = f'DIV2K_train_HR/{image_id}.png'
            info = archive.getinfo(member)
            if info.file_size > 32*1024*1024:
                raise RuntimeError('Unexpectedly large image.')
            path = output/f'{image_id}.png'
            existed = path.exists()
            if existed:
                payload = path.read_bytes()
            else:
                for attempt in range(3):
                    try:
                        payload = archive.read(member)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if attempt == 2:
                            raise
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                raise RuntimeError(f'Archive CRC/length mismatch for {image_id}.')
            if image_id in known and known[image_id]['sha256'] != digest:
                raise RuntimeError(f'Local SHA-256 mismatch for {image_id}.')
            if not existed:
                temporary = path.with_suffix('.png.part')
                temporary.write_bytes(payload)
                temporary.replace(path)
            records.append({'id': image_id, 'project_split': group, 'archive_member': member,
                            'path': str(path.relative_to(ROOT)), 'sha256': digest,
                            'bytes': len(payload), 'zip_crc32': f'{info.CRC:08x}'})
            report = {'status': 'complete' if index == len(selected) else 'partial',
                      'archive_url': URL, 'archive_etag': remote.etag, 'archive_bytes': remote.length,
                      'candidate_manifest_sha256': hashlib.sha256(split_bytes).hexdigest(),
                      'transferred_bytes_this_run': remote.downloaded,
                      'raw_bytes_acquired': sum(r['bytes'] for r in records), 'files': records}
            temporary = record_path.with_suffix('.json.part')
            temporary.write_text(json.dumps(report, indent=2)+'\n')
            temporary.replace(record_path)
            if index % 10 == 0 or index == 1:
                print(f'{index}/{len(selected)} checked and available; {report["raw_bytes_acquired"]/1e6:.1f} MB', flush=True)
    print('Acquisition complete. No restoration performance was evaluated on test images.')


if __name__ == '__main__':
    main()
