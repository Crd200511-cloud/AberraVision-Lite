"""Offline demonstration with a saved model and a self-generated scene."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.model import ResidualCNN
from src.optics import simulate_observation
from src.baseline import wiener_restore
from src.patterns import learning_pattern


def run_demo(output: Path=ROOT/'results/demo', checkpoint: Path=ROOT/'models/aberravision_lite.pt'):
    torch.set_num_threads(4)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=ResidualCNN().eval(); model.load_state_dict(saved['state_dict'])
    protocol=saved['provenance']['protocol']
    target=learning_pattern(384)
    sample=simulate_observation(target,**protocol['optics'],noise_sigma=protocol['noise_sigma'],seed=20261013)
    baseline=wiener_restore(sample['noisy'],sample['psf'],saved['provenance']['selected_k'])
    with torch.inference_mode():
        predicted=model(torch.from_numpy(np.stack([sample['noisy'],baseline]).astype(np.float32))[None])[0,0].numpy()
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,4,figsize=(12,3.6),layout='constrained')
    for ax,array,title in zip(axes,[target,sample['noisy'],baseline,predicted],['Synthetic original','Blur + noise','Wiener','Wiener + CNN']):
        ax.imshow(array[64:320,64:320],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(title,fontsize=11); ax.set_axis_off()
    fig.suptitle('Saved-model demo | Self-generated pattern | Not the natural-photo benchmark',fontsize=11)
    figure=output/'demo.png'; fig.savefig(figure,dpi=150); plt.close(fig)
    metadata={'purpose':'self-generated demonstration, not a benchmark or extra training condition',
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'checkpoint_epoch':saved['epoch'],'noise_seed':20261013,
              'optics':sample['metadata'],'wiener_k':saved['provenance']['selected_k'],
              'display_range':[0,1],'display_crop_ltrb':[64,64,320,320]}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return figure


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/demo')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/aberravision_lite.pt')
    args=parser.parse_args(); print(run_demo(args.output,args.checkpoint))
