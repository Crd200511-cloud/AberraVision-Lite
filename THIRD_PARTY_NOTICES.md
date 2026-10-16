# Sources and third-party material

## DIV2K

Source: https://data.vision.ee.ethz.ch/cvl/DIV2K/

The official page provides the dataset for academic research and states that image copyrights remain with the original owners. The repository excludes the downloaded images, preprocessed image arrays, photographic contact sheets, and photographic result figures. Data scripts obtain the selected images from the official archive. Do not apply a source-code license to these photographs or assume unrestricted redistribution rights.

Please cite:

- Eirikur Agustsson and Radu Timofte. **NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.** CVPR Workshops, 2017.
- Radu Timofte et al. **NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results.** CVPR Workshops, 2017.

This project defines its own split inside the official training HR collection. Its scores are not DIV2K challenge results.

## Software and implementation references

- NumPy, SciPy, Matplotlib, Pillow, PyTorch, scikit-image, and Jupyter tools retain their respective upstream licenses. They are installed as dependencies, not bundled source copies.
- [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html), [Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html), and [serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html).
- [scikit-image PSNR/SSIM](https://scikit-image.org/docs/stable/api/skimage.metrics.html).
- Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. **Image Quality Assessment: From Error Visibility to Structural Similarity.** IEEE Transactions on Image Processing, 2004. https://doi.org/10.1109/TIP.2003.819861
- [POPPY wavefront-error reference](https://poppy-optics.readthedocs.io/en/latest/wfe.html), for background reading; POPPY is not a dependency and is not claimed as an independent numerical validator here.

## Generated artifacts

The synthetic stripe/edge pattern and corresponding demonstration images are programmatically generated in this project. Trained weights are derived from academic-research data and are provided as a research/educational artifact; this project does not assert unrestricted commercial rights in upstream data. No image-generation model is used to fabricate experimental results.

No source-code license has yet been selected by the project owner. A public GitHub repository alone does not grant a reuse license; choose and add a license before representing the code as openly licensed.
