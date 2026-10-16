"""Prepare scene proxies and screen duplicates; never score restorations."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from src.data import prepare_scene
from src.data_audit import perceptual_signatures, hash_distance


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2)+'\n')


def main():
    record_dir = ROOT/'results/data_preparation'
    acquisition = json.loads((record_dir/'dataset_acquisition.json').read_text())
    if acquisition['status'] != 'complete':
        raise RuntimeError('Acquire all selected images before preparing the dataset.')
    candidate_path = record_dir/'candidate_split.json'
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != acquisition['candidate_manifest_sha256']:
        raise RuntimeError('Candidate split changed after acquisition.')
    candidate = json.loads(candidate_path.read_text())
    expected = {(group, image_id) for group, ids in candidate['splits'].items() for image_id in ids}
    actual = [(r['project_split'], r['id']) for r in acquisition['files']]
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError('Acquisition does not match the source-image split.')
    output = ROOT/'data/prepared'
    output.mkdir(parents=True, exist_ok=True)
    audit_images = ROOT/'results/data_preview/audit'
    audit_images.mkdir(parents=True, exist_ok=True)
    records, thumbnails = [], {}
    for index, item in enumerate(acquisition['files'], start=1):
        path = ROOT/item['path']
        scene, preprocessing = prepare_scene(path, size=384)
        if preprocessing['source_sha256'] != item['sha256']:
            raise RuntimeError(f'Source SHA mismatch: {item["id"]}')
        if not np.isfinite(scene).all() or np.min(scene) < 0 or np.max(scene) > 1:
            raise RuntimeError('Invalid prepared scene.')
        scene32 = scene.astype(np.float32)
        scene_path = output/f'{item["id"]}.npy'
        np.save(scene_path, scene32, allow_pickle=False)
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            pixel_digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            signatures = perceptual_signatures(rgb)
            thumbnails[item['id']] = ImageOps.contain(rgb, (180,135))
        scene_signatures = perceptual_signatures(Image.fromarray(np.round(scene32*255).astype(np.uint8)))
        records.append({**item, 'preprocessing': preprocessing, 'scene_path': str(scene_path.relative_to(ROOT)),
                        'scene_file_sha256': hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                        'decoded_rgb_sha256': pixel_digest,
                        'scene_pixels_sha256': hashlib.sha256(scene32.tobytes()).hexdigest(),
                        'full_image_phash': signatures, 'scene_phash': scene_signatures})
        if index % 25 == 0:
            print(f'{index}/150 preprocessed and fingerprinted', flush=True)
    pairs = []
    for first, second in itertools.combinations(records, 2):
        exact = any(first[field] == second[field] for field in
                    ('sha256', 'decoded_rgb_sha256', 'scene_pixels_sha256'))
        full_distance = hash_distance(first['full_image_phash'], second['full_image_phash'])
        scene_distance = hash_distance(first['scene_phash'], second['scene_phash'])
        if exact or min(full_distance, scene_distance) <= 8:
            pairs.append({'ids': [first['id'], second['id']],
                          'splits': [first['project_split'], second['project_split']],
                          'exact_duplicate': exact, 'full_hash_distance': full_distance,
                          'scene_hash_distance': scene_distance, 'review': 'pending'})
    # Contact sheets show source content only, never a test reconstruction or score.
    for page, start in enumerate(range(0, len(records), 50), start=1):
        canvas = Image.new('RGB', (5*190, 10*160), 'white')
        draw = ImageDraw.Draw(canvas)
        for offset, item in enumerate(records[start:start+50]):
            x, y = offset%5*190, offset//5*160
            canvas.paste(thumbnails[item['id']], (x,y+20))
            draw.text((x,y+3), f'{item["id"]} / {item["project_split"]}', fill='black')
        canvas.save(audit_images/f'sources_{page}.jpg', quality=90)
    for page, start in enumerate(range(0, len(pairs), 10), start=1):
        canvas = Image.new('RGB', (520, 10*160), 'white')
        draw = ImageDraw.Draw(canvas)
        for offset, pair in enumerate(pairs[start:start+10]):
            y = offset*160
            draw.text((0,y), f'{start+offset}: {pair["ids"]} / distances {pair["full_hash_distance"]},{pair["scene_hash_distance"]}', fill='black')
            for col, image_id in enumerate(pair['ids']):
                canvas.paste(thumbnails[image_id], (col*260,y+20))
        canvas.save(audit_images/f'flagged_pairs_{page}.jpg', quality=90)
    save_json(record_dir/'prepared_index.json', {'status': 'prepared_pending_duplicate_review',
              'candidate_manifest_sha256': acquisition['candidate_manifest_sha256'],
              'scene_size': 384, 'scene_storage_dtype': 'float32', 'files': records})
    save_json(record_dir/'duplicate_screening.json', {
        'status': 'pending_visual_review', 'source_count': len(records),
        'exact_checks': 'source bytes, oriented decoded RGB, preprocessed scene pixels',
        'near_duplicate_screen': '63-bit DCT pHash <= 8 on full image or center scene; rotations and mirrors',
        'limitations': 'Heuristic screening may miss recrops, different exposures, or other same-scene images.',
        'restoration_test_scores_computed': False, 'pairs': pairs})
    print(f'Prepared {len(records)} images; {len(pairs)} pairs flagged for visual review.')


if __name__ == '__main__':
    main()
