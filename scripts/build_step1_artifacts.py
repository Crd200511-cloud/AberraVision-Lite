"""Rebuild the executed lesson notebook and numerical evidence for stage 1."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache" / "ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".cache" / "jupyter"))

import nbformat as nbf
import numpy as np
from PIL import Image
from scipy.special import j1

from src.optics import make_psf_with_info, circular_convolve, simulate_observation
from src.patterns import learning_pattern


def png_uri(array):
    buf = io.BytesIO()
    Image.fromarray(np.round(np.clip(array, 0, 1)*255).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def make_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "AberraVision (.venv)", "language": "python", "name": "aberravision"}
    nb.cells = [
        nbf.v4.new_markdown_cell("# 第 1 步：失焦与噪声\n\n只做光学成像，不进行 AI 复原。你的实验前预测是：**边缘条纹更加模糊**。\n\n每次只改一个参数，然后重新运行参数格及后面的格子。实际观察写入 `PARTICIPATION.md`。"),
        nbf.v4.new_code_cell("from pathlib import Path\nimport sys\nROOT = Path.cwd()\nif not (ROOT / 'src').exists():\n    raise RuntimeError('请从项目根目录运行这个 Notebook')\nsys.path.insert(0, str(ROOT))\nfrom IPython.display import Image, display\nfrom learn_optics import run_experiment\nprint('项目路径:', ROOT)"),
        nbf.v4.new_markdown_cell("## 你修改这一格\n\n实验 A：保持 `NOISE_SIGMA = 0.01`，依次尝试失焦 `0.0`、`0.15`、`0.30`。\n\n实验 B：保持 `DEFOCUS_WAVES = 0.15`，依次尝试噪声 `0.0`、`0.01`、`0.03`。`SEED` 固定，保证比较使用同一个底层随机样本。"),
        nbf.v4.new_code_cell("DEFOCUS_WAVES = 0.15\nNOISE_SIGMA = 0.01\nSEED = 42"),
        nbf.v4.new_code_cell("result = run_experiment(DEFOCUS_WAVES, NOISE_SIGMA, SEED)\ndisplay(Image(filename=str(ROOT / 'results/learning/current_comparison.png')))"),
        nbf.v4.new_markdown_cell("四个面板依次是原图、光斑 PSF、仅模糊、模糊加噪。PSF 使用固定对数灰度以显示较暗的结构；场景图均使用固定 `[0,1]` 显示范围。\n\n观察细条纹的明暗差、斜边过渡和左下方平滑区域。零失焦仍存在有限孔径衍射。"),
        nbf.v4.new_code_cell("print('PSF 总和:', result['psf'].sum())\nprint('截取区域保留的离散光能比例:', result['metadata']['crop_energy_fraction'])\nprint('像面采样间距:', result['metadata']['q_step'], 'lambda*f/D')\nprint('原始观测范围:', result['noisy'].min(), result['noisy'].max())\nprint('是否截断过观测:', result['metadata']['observation_clipped'])"),
        nbf.v4.new_markdown_cell("## 完成本步前，写下自己的观察\n\n1. 失焦增大后，你实际看到哪处细节变化？\n2. 只改噪声时，PSF 是否变化？\n3. 这与你的预测是否一致？\n\n这些答案需要你填写。学习参数不自动成为正式实验参数；当前图片也不是 AI 复原结果。")
    ]
    path = ROOT / "01_optics_learning.ipynb"
    nbf.write(nb, path)
    return path


def save_validation():
    out = ROOT / "results/validation"
    out.mkdir(parents=True, exist_ok=True)
    psf, info = make_psf_with_info()
    airy, ai = make_psf_with_info(0, 128, 2048, 129)
    c = len(airy)//2
    q = np.arange(c+1)*ai['q_step']
    analytic = np.ones_like(q)
    analytic[1:] = (2*j1(np.pi*q[1:])/(np.pi*q[1:]))**2
    numerical = airy[c,c:]/airy[c,c]
    indices = np.flatnonzero((q > .9) & (q < 1.5))
    measured_min = float(q[indices[np.argmin(numerical[indices])]])
    refined, refined_info = make_psf_with_info(.15, 256, 1024, 129)
    extended, extended_info = make_psf_with_info(.15, 128, 512, 257)
    scene = np.zeros((320,320)); scene[80:240,80:240] = 1
    metrics = {
        "purpose": "stage_1_numerical_checks_not_reconstruction_benchmark",
        "default_psf": info,
        "psf_sum": float(psf.sum()),
        "airy_max_profile_error": float(np.max(np.abs(numerical-analytic))),
        "airy_first_dark_ring_q_numerical": measured_min,
        "airy_first_dark_ring_q_analytic": 1.21966989,
        "airy_validation_q_step": ai['q_step'],
        "refined_q_step": refined_info['q_step'],
        "grid_refinement_psf_l1": float(np.abs(psf-refined).sum()),
        "grid_refinement_image_max_abs": float(np.abs(circular_convolve(scene,psf)-circular_convolve(scene,refined)).max()),
        "extended_kernel_energy_fraction": extended_info['crop_energy_fraction'],
        "kernel_extension_image_max_abs": float(np.abs(circular_convolve(scene,psf)-circular_convolve(scene,extended)).max()),
        "note": "Tolerances are documented in tests/test_optics.py. Crop energy is relative to a finite FFT grid, not the infinite optical plane."
    }
    (out/'optics_checks.json').write_text(json.dumps(metrics, indent=2)+"\n")


def preview_payload():
    image = learning_pattern(192)
    zero,_ = make_psf_with_info(0)
    peak = float(zero.max())
    payload = {"original": png_uri(image), "defocus": [0, .15, .3], "noise": [0, .01, .03], "cases": []}
    for a in payload['defocus']:
        row = []
        for sigma in payload['noise']:
            result = simulate_observation(image, defocus_waves=a, noise_sigma=sigma, seed=42)
            h = result['psf']; c = h.shape[0]//2
            display_psf = (np.log10(np.maximum(h[c-32:c+33,c-32:c+33]/peak,1e-5))+5)/5
            row.append({"observed": png_uri(result['noisy']), "psf": png_uri(display_psf),
                        "retained": round(result['metadata']['crop_energy_fraction']*100,3)})
        payload['cases'].append(row)
    dest = ROOT/'results/learning/preview_data.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, separators=(',',':')))
    return dest


if __name__ == '__main__':
    print(make_notebook())
    save_validation()
    print(preview_payload())
