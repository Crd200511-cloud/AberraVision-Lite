"""Evaluate the frozen checkpoint on all 30 held-out source photos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch

from src.baseline import wiener_restore
from src.metrics import score_reconstruction, SSIM_SETTINGS
from src.model import ResidualCNN
from src.optics import simulate_observation


def make_comparison(image_id, arrays, output):
    fig, axes = plt.subplots(2,4,figsize=(12,6.8),layout='constrained')
    crop=np.s_[64:320,64:320]
    target=arrays['original'][crop]
    methods=['original','observed','wiener','cnn']
    titles=['Original target','Blur + noise','Wiener','Wiener + CNN']
    for col,(name,title) in enumerate(zip(methods,titles)):
        view=np.clip(arrays[name][crop],0,1)
        axes[0,col].imshow(view,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        axes[0,col].set_title(title)
        if name=='original':
            axes[1,col].text(.05,.5,'Absolute errors below\nSame scale: 0 to 0.2\nBrighter = larger error',transform=axes[1,col].transAxes)
        else:
            im=axes[1,col].imshow(np.abs(view-target),cmap='magma',vmin=0,vmax=.2,interpolation='nearest')
        axes[0,col].set_axis_off(); axes[1,col].set_axis_off()
    fig.colorbar(im,ax=list(axes[1,1:]),shrink=.8,label='Absolute error (saturates above 0.2)')
    fig.suptitle(f'Held-out source {image_id} | Fixed center 256x256 | Same observation for all methods')
    fig.savefig(output/f'{image_id}_comparison.png',dpi=150)
    plt.close(fig)


def main(checkpoint_path, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; preserve prior test results.')
    protocol_bytes=(ROOT/'config.json').read_bytes()
    protocol=json.loads(protocol_bytes)
    manifest_bytes=(ROOT/'results/data_preparation/data_manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    if hashlib.sha256(manifest_bytes).hexdigest()!=protocol['data_manifest_sha256']:
        raise ValueError('Data manifest changed after freezing.')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['config'].get('protocol_sha256')!=hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError('Checkpoint and frozen protocol differ.')
    if checkpoint['provenance']['data_manifest_sha256']!=protocol['data_manifest_sha256']:
        raise ValueError('Checkpoint used different data.')
    if checkpoint['provenance']['wiener_selection_sha256']!=protocol['wiener_selection_sha256']:
        raise ValueError('Checkpoint used a different Wiener selection.')
    if protocol['evaluation']['ssim']!=SSIM_SETTINGS or protocol['evaluation']['score_crop_ltrb']!=[64,64,320,320]:
        raise ValueError('Scoring implementation differs from frozen protocol.')
    training_summary=json.loads((checkpoint_path.parent/'summary.json').read_text())
    if not training_summary['formal_training_completed'] or checkpoint['epoch']!=training_summary['best_epoch']:
        raise ValueError('Evaluate only the selected checkpoint of the completed formal run.')
    records=[r for r in manifest['files'] if r['project_split']=='test']
    if len(records)!=protocol['evaluation']['test_count']:
        raise ValueError('Unexpected test source count.')
    seen=set(sum(checkpoint['provenance']['source_ids'].values(),[]))
    if any(r['id'] in seen for r in records):
        raise ValueError('Test source leaked into training or model selection.')
    torch.set_num_threads(4)
    model=ResidualCNN().eval()
    model.load_state_dict(checkpoint['state_dict'])
    optics=checkpoint['provenance']['protocol']['optics']
    sigma=checkpoint['provenance']['protocol']['noise_sigma']
    k=checkpoint['provenance']['selected_k']
    output.mkdir(parents=True,exist_ok=True)
    raw_dir=ROOT/'results/test_arrays'/output.name
    raw_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for record in records:
        path=ROOT/record['scene_path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['scene_file_sha256']:
            raise ValueError('Prepared test scene checksum mismatch.')
        target=np.load(path,allow_pickle=False).astype(np.float64)
        sample=simulate_observation(target,**optics,noise_sigma=sigma,seed=100000+int(record['id']))
        baseline=wiener_restore(sample['noisy'],sample['psf'],k)
        inputs=np.stack([sample['noisy'],baseline]).astype(np.float32)
        with torch.inference_mode():
            predicted=model(torch.from_numpy(inputs)[None])[0,0].numpy().astype(np.float64)
        arrays={'original':target,'observed':sample['noisy'],'wiener':baseline,'cnn':predicted}
        # Float64 baseline retained; network is float32, as recorded at training.
        row={'id':record['id']}
        for name in ('observed','wiener','cnn'):
            row.update({f'{name}_{metric}':value for metric,value in score_reconstruction(target,arrays[name]).items()})
        row['delta_psnr_db']=row['cnn_psnr_db']-row['wiener_psnr_db']
        row['delta_ssim']=row['cnn_ssim']-row['wiener_ssim']
        rows.append(row)
        np.savez_compressed(raw_dir/f'{record["id"]}.npz',**arrays)
    with (output/'per_image.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    means={name:{metric:float(np.mean([r[f'{name}_{metric}'] for r in rows]))
                 for metric in ('psnr_db','ssim','l1')} for name in ('observed','wiener','cnn')}
    worst=min(rows,key=lambda r:(r['delta_psnr_db'],r['id']))
    fixed=protocol['evaluation']['fixed_example_ids']
    examples=list(dict.fromkeys(fixed+[worst['id']]))
    summary={'test_source_count':len(rows),'checkpoint_epoch':checkpoint['epoch'],
             'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
             'protocol_sha256':hashlib.sha256(protocol_bytes).hexdigest(),
             'data_manifest_sha256':protocol['data_manifest_sha256'],
             'metrics_version':skimage.__version__,'numpy_version':np.__version__,
             'evaluation_code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                       for name in ['evaluate.py','src/metrics.py']},
             'means':means, 'mean_delta_psnr_db':float(np.mean([r['delta_psnr_db'] for r in rows])),
             'mean_delta_ssim':float(np.mean([r['delta_ssim'] for r in rows])),
             'psnr_improved_count':sum(r['delta_psnr_db']>0 for r in rows),
             'ssim_improved_count':sum(r['delta_ssim']>0 for r in rows),
             'psnr_worsened_count':sum(r['delta_psnr_db']<0 for r in rows),
             'ssim_worsened_count':sum(r['delta_ssim']<0 for r in rows),
             'worst_psnr_delta_example':worst,'fixed_examples':fixed,'displayed_examples':examples,
             'scoring':protocol['evaluation'], 'single_training_run':True,
             'test_outcomes_used_to_change_method':False}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figure_dir=ROOT/'results/data_preview/test'/output.name
    figure_dir.mkdir(parents=True,exist_ok=True)
    for image_id in examples:
        with np.load(raw_dir/f'{image_id}.npz') as arrays:
            make_comparison(image_id,arrays,figure_dir)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ordered=sorted(rows,key=lambda r:r['id'])
    for ax,metric,label in zip(axes,['delta_psnr_db','delta_ssim'],['PSNR change (dB)','SSIM change']):
        values=[r[metric] for r in ordered]
        ax.bar(range(30),values,color=['#247a69' if v>=0 else '#bd522a' for v in values])
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_xticks(range(30),[r['id'] for r in ordered],rotation=90,fontsize=7)
        ax.set(xlabel='Held-out source ID',ylabel=label,title='CNN minus Wiener; positive = improvement')
    fig.savefig(output/'paired_changes.png',dpi=150); plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'results/formal/run_01/best.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'results/test/run_01')
    args=parser.parse_args()
    main(args.checkpoint,args.output)
