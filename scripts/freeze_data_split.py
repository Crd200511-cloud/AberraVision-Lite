"""Freeze source IDs after an explicit duplicate review record is supplied."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT/'results/data_preparation'
    index_bytes = (directory/'prepared_index.json').read_bytes()
    screening_bytes = (directory/'duplicate_screening.json').read_bytes()
    review = json.loads((directory/'duplicate_review.json').read_text())
    if review['prepared_index_sha256'] != hashlib.sha256(index_bytes).hexdigest():
        raise ValueError('Review does not match prepared data.')
    if review['screening_sha256'] != hashlib.sha256(screening_bytes).hexdigest():
        raise ValueError('Review does not match duplicate screening.')
    if review['status'] != 'review_complete' or review['unresolved_pairs']:
        raise ValueError('Resolve duplicate review first.')
    screening = json.loads(screening_bytes)
    if screening['pairs']:
        raise ValueError('This minimal freeze script requires a screening with no flagged pairs; resolve and document flagged data first.')
    index = json.loads(index_bytes)
    by_id = {r['id']: r for r in index['files']}
    for group in review['related_source_groups']:
        if len({by_id[i]['project_split'] for i in group['ids']}) != 1:
            raise ValueError('Related source content crosses data splits.')
    manifest = {**index, 'status': 'source_split_frozen_after_duplicate_review',
                'duplicate_review': review,
                'counts': {name: sum(r['project_split']==name for r in index['files'])
                           for name in ('train','validation','test')}}
    target = directory/'data_manifest.json'
    text = json.dumps(manifest, indent=2)+'\n'
    if target.exists() and target.read_text() != text:
        raise ValueError('A different frozen manifest already exists.')
    target.write_text(text)
    print(manifest['counts'])


if __name__ == '__main__':
    main()
