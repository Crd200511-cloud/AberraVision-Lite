"""Fetch two train candidates from the official archive using HTTP ranges.

No full archive download, test-image fetch, or archive path extraction.
The manifest is provisional until a content duplicate audit is completed.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import candidate_split

URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'


class RemoteZip(io.RawIOBase):
    """Seekable read-only HTTP range stream with bounded, validated responses."""

    def __init__(self, url: str):
        self.url, self.position, self.downloaded = url, 0, 0
        self.etag = None
        tail, start, self.length = self._request('bytes=-65557', 65557)
        self.tail, self.tail_start = tail, start

    def _request(self, byte_range: str, maximum: int):
        headers = {'Range': byte_range, 'Accept-Encoding': 'identity'}
        if self.etag:
            headers['If-Match'] = self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=45) as response:
            if response.status != 206:
                raise RuntimeError('Server ignored Range; refusing full-archive download.')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match:
                raise RuntimeError('Invalid Content-Range.')
            start, end, total = map(int, match.groups())
            if end-start+1 > maximum:
                raise RuntimeError('Server returned an oversized range.')
            if byte_range.startswith('bytes=-'):
                if start != max(0, total-maximum) or end != total-1:
                    raise RuntimeError('Incorrect suffix range.')
            elif byte_range != f'bytes={start}-{end}':
                raise RuntimeError('Incorrect byte range.')
            etag = response.headers.get('ETag')
            if self.etag and etag != self.etag:
                raise RuntimeError('Archive changed during reading.')
            self.etag = etag
            payload = response.read(maximum+1)
            if len(payload) != end-start+1:
                raise RuntimeError('Incomplete range response.')
        self.downloaded += len(payload)
        return payload, start, total

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('Invalid whence.')
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.length)
        if position < 0:
            raise ValueError('Negative seek.')
        self.position = position
        return position

    def read(self, size=-1):
        available = max(0, self.length-self.position)
        size = available if size is None or size < 0 else min(size, available)
        if size > 32*1024*1024:
            raise RuntimeError('Unexpected read above 32 MiB.')
        if size == 0:
            return b''
        if self.position >= self.tail_start:
            start = self.position-self.tail_start
            payload = self.tail[start:start+size]
        else:
            payload, _, total = self._request(f'bytes={self.position}-{self.position+size-1}', size)
            if total != self.length:
                raise RuntimeError('Archive length changed.')
        self.position += len(payload)
        return payload


def main():
    splits = candidate_split()
    preview_ids = splits['train'][:2]
    output = ROOT/'data/raw/DIV2K_train_HR'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with RemoteZip(URL) as remote, zipfile.ZipFile(remote) as archive:
        for image_id in preview_ids:
            member = f'DIV2K_train_HR/{image_id}.png'
            info = archive.getinfo(member)
            if info.file_size > 32*1024*1024:
                raise RuntimeError('Unexpected image size.')
            # zipfile validates each member CRC while reading to completion.
            payload = archive.read(member)
            path = output/f'{image_id}.png'
            digest = hashlib.sha256(payload).hexdigest()
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Existing local file differs: {path}')
            path.write_bytes(payload)
            records.append({'id': image_id, 'project_split': 'train', 'archive_member': member,
                            'path': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': len(payload)})
            print(f'Fetched training candidate {image_id}: {len(payload)/1e6:.1f} MB', flush=True)
        acquisition = {'archive_url': URL, 'archive_etag': remote.etag, 'archive_bytes': remote.length,
                       'transferred_bytes': remote.downloaded, 'files': records}
    metadata_dir = ROOT/'results/data_preparation'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'candidate_split_pending_full_acquisition_and_duplicate_audit',
                'source': 'DIV2K official train HR 0001-0800', 'seed': 20261012,
                'split_type': 'project-specific, not official challenge benchmark',
                'splits': splits, 'preview_ids': preview_ids}
    (metadata_dir/'candidate_split.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (metadata_dir/'preview_acquisition.json').write_text(json.dumps(acquisition, indent=2)+'\n')
    print(f'Total transferred: {acquisition["transferred_bytes"]/1e6:.1f} MB')


if __name__ == '__main__':
    main()
