"""Build the compact report from saved formal results, without rerunning tests."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    test=json.loads((ROOT/'results/test/run_01/summary.json').read_text())
    train=json.loads((ROOT/'results/formal/run_01/summary.json').read_text())
    config=json.loads((ROOT/'config.json').read_text())
    lines=['| 方法 | 平均 PSNR（dB）↑ | 平均 SSIM ↑ | 平均 L1 ↓ |','|---|---:|---:|---:|']
    for name,label in [('observed','模糊观测'),('wiener','Wiener'),('cnn','Wiener＋CNN')]:
        row=test['means'][name]
        lines.append(f'| {label} | {row["psnr_db"]:.4f} | {row["ssim"]:.5f} | {row["l1"]:.5f} |')
    table='\n'.join(lines)
    worst=test['worst_psnr_delta_example']
    outcome='提高' if test['mean_delta_psnr_db']>0 else '降低'
    report=f'''# AberraVision Lite：固定失焦条件下的 AI 辅助图像复原

日期：2026-10-13。项目性质：AI 助手协作的教育性计算研究。无真实光学硬件实验，未声称提出新网络架构或已发表论文。实际个人参与见 `PARTICIPATION.md`。

## 摘要

本项目研究一个范围有限的问题：固定失焦与噪声条件下，在 Wiener 反卷积后加入轻量残差 CNN，是否改善未用于训练和选参的照片复原质量。用 DIV2K 中 150 张照片构造线性灰度场景代理，按原图分为 100 张训练、20 张验证、30 张测试。光学系统采用圆孔、标量单色非相干模型；所有图片使用同一 PSF 和噪声强度。CNN 共 28,641 参数，进行一次固定预算训练。

在这 30 张测试照片上，CNN 相对 Wiener 的平均 PSNR {outcome} {abs(test['mean_delta_psnr_db']):.4f} dB，平均 SSIM 改变量为 {test['mean_delta_ssim']:+.5f}。其中 PSNR 改善 {test['psnr_improved_count']}/30 张、退步 {test['psnr_worsened_count']}/30 张；SSIM 改善 {test['ssim_improved_count']}/30 张、退步 {test['ssim_worsened_count']}/30 张。结果只适用于本批场景及所测固定合成条件。

## 1. 研究问题与方法依据

光学模糊削弱细节；反卷积补偿这种削弱时也可能放大噪声。用户在教学实验中观察到细条纹先变得难以分清，增加噪声时 PSF 不变，Wiener 复原后的天空噪点更明显；调大 K 后总结为“减少噪点，但也会让画面变模糊”。这些观察引出降噪与细节保留的权衡。

我们使用已知 PSF 的 Wiener 作为唯一传统基线，再让小型 CNN 学习修正剩余误差。设计是现有光学与机器学习组件的组合与受控比较，不是新算法的优先权声明。没有预设 AI 必然优于基线。

## 2. 成像模型

单位圆孔内失焦模式为 Z(ρ)=√3(2ρ²−1)，复光瞳 P=A exp(i2πaZ)。PSF 由 Fourier 变换强度 |F(P)|² 得到，再归一化。采用 a=0.15 RMS waves、光瞳直径 128 个采样间隔、FFT 边长 512、PSF 核宽 129。归一化像面采样间隔为 0.25 λf/D；不映射特定镜头或真实传感器。

观测 y=h*x+n，其中卷积为显式周期卷积，n 为零均值独立高斯噪声，σ=0.01。场景经 sRGB 逆传递与线性亮度加权得到，缩小并中心裁成 384×384。自然照片本身包含摄影与后期处理，因此仅是合成物体纹理代理，不是 RAW 光学测量。原始浮点观测不截断。

必要检查包括无像差 Airy 形状与首暗环、非负与归一化、冲激响应、直接周期卷积与 FFT 一致，以及采样和核范围稳定性。129 核保留约 98.864% 的有限 FFT 网格能量；重新归一化不能消除核裁切近似。训练样本上加密光瞳与扩展 PSF 的变化通过事先规定的数值容差，完整记录见 `results/validation/`。

## 3. 基线、网络与训练

Wiener 使用 X_W=conj(H)Y/(|H|²+K)，对应常数噪声／信号频谱比近似或单位算子 Tikhonov。它不是默认 Laplacian 惩罚的 Wiener–Hunt。K 在 0.0001、0.001、0.003、0.01、0.03 中按 20 张验证图的平均 L1 选择，结果为 0.01。CNN 前处理共享同一实现与 K。

网络为 x_hat=x_W+fθ(y,x_W)，五层 3×3 卷积，通道 2→32→32→32→32→1，前四层接 ReLU，无 BN、注意力、U-Net 或额外物理损失。末层零初始化，使初始输出等于 Wiener。输入只有 y 与 x_W；原图 x 仅用于监督和评价。

正式训练从头初始化，种子 20261013，CPU 4 线程，Adam 学习率 0.0005，固定 30 轮。每图每轮随机取 4 个 128×128 有效块，每批 4 块，共 3000 次更新。裁块在完整图退化及 Wiener 之后生成，额外取四周 5 像素上下文，以免填充边缘进入训练损失。训练使用未截断输出的 L1。

训练和验证监测均对完整图推理，统一在中央 256×256 区域截断至 [0,1] 后计分。按验证 L1 最低保留第 **{train['best_epoch']}** 轮，验证 L1 为 **{train['best_validation_l1']:.6f}**。训练与监测共约 {train['total_train_and_monitor_seconds']/60:.2f} 分钟。权重重新加载后输出核查通过。正式训练之前做过 8/4 张照片、12 轮的小样本调试，未将其权重用于正式初始化；不能把整个开发过程描述成只有一次运行。

## 4. 数据划分与评价

150 个来源 ID 按固定种子从 DIV2K 官方训练 HR 集合生成项目自定义划分，不属于官方挑战测试协议。先分原图，再裁块。文件、解码像素及场景像素哈希，加上旋转／镜像感知哈希与来源缩略图核查，未发现明显跨组重复。0207 和 0655 是同一博物馆展品的不同视角，均留在训练集。该筛查不能保证识别所有同场景来源。

最终测试共 30 张原图，所有方法共享观测和 PSF，均以中央 256×256 计分，并统一将预测截到 [0,1]。PSNR 的 data_range=1；SSIM 使用 11×11 Gaussian 窗、σ=1.5、总体协方差、K1=0.01、K2=0.03。先逐原图计分再取算术平均，另保存配对改变量。SSIM 的窗口平均由库在裁图有效位置计算。

测试不参与参数、轮数、模型选择，也不在看到结果后追加训练。未进行多随机种子验证、显著性检验或跨条件泛化实验。

## 5. 结果

{table}

![逐图改变量](results/test/run_01/paired_changes.png)

PSNR 与 SSIM 衡量不同方面，平均值也不保证每张图都改善。逐图数据在 `results/test/run_01/per_image.csv`，检查范围包括改善与退步，不仅展示成功图片。

预先固定展示图为 0189、0336。按事先定义的“PSNR 改变量最小”规则，该案例为 **{worst['id']}**：ΔPSNR={worst['delta_psnr_db']:+.4f} dB，ΔSSIM={worst['delta_ssim']:+.5f}。它恰好与固定展示图 0336 重合，因此合并为两组不同照片的对比，不在看过测试结果后另挑第三张。全部 30 张在两项指标上均改善，所以本次没有捏造退步案例。

助手查看原图、输出与同范围误差图后观察到：0189 的眼部轮廓与较粗毛发结构比 Wiener 更鲜明，但细密毛发仍有融合和形状偏差；0336 的背景噪纹减弱，脸部轮廓较清楚，但鼻口与脸颊的细毛并未恢复成原图那样的纹理，仍呈现较平滑或成块的结构。误差图也显示毛发和轮廓处仍有残差。这些是助手的图像观察，不能写成用户已自行得出的发现，也不能从观感推断所有细节都真实恢复。

用户在查看 0336 后反馈：“没有完整恢复，照片的像素质量还差很多”。这支持其对复原仍不完整的判断；“细毛融合成块、像素尺寸不变”的具体术语说明由助手补充。照片对比及统一范围误差图保存在本地 `results/data_preview/test/run_01/`，不将第三方照片默认为可自由再发布素材。原始观察与协助记录见 `PARTICIPATION.md`。

## 6. 局限与结论

证据限于固定失焦、固定噪声、纯合成退化、空间不变 PSF 和周期边界。有限 PSF 核有尾部损失，中心计分只减弱边缘影响。训练与测试使用同一模拟器，无法代表与真实镜头模型不匹配时的表现。仅一次正式训练，无法声称对随机性稳定；数据量较小，来源独立性筛查仍有局限。

所得结论是：在本次固定协议和 30 张独立保留照片上，Wiener＋CNN 相对 Wiener 的平均 PSNR 改变量为 {test['mean_delta_psnr_db']:+.4f} dB，平均 SSIM 改变量为 {test['mean_delta_ssim']:+.5f}。不据此声称突破衍射极限、恢复必然真实的高频细节、适用于未知相机或普遍优于传统算法。

## 7. 复现与贡献

固定参数见 `config.json`，运行说明见 `README.md`。`demo.ipynb` 默认读取已保存模型，用自生成图案演示完整流程；它不是自然照片测试集。源码、配置、数据 ID、校验哈希、环境、权重、日志与逐图分数共同支持复现。

AI 助手大量协助代码、推导说明、实验执行、分析与写作。用户提出预测、反馈失焦与噪声观察、选择 K 调整方向、总结降噪与模糊的权衡，并参与 L1 与曲线阅读学习。用户尚未实际完成的操作不列为已完成贡献，详见 `PARTICIPATION.md`。将本项目用于申请时，应如实描述这种协作方式。

## 参考

- Agustsson, E. & Timofte, R. (2017). NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. CVPR Workshops. [DIV2K 官方入口](https://data.vision.ee.ethz.ch/cvl/DIV2K/)。
- Timofte, R. et al. (2017). NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results. CVPR Workshops.
- Wang, Z. et al. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE TIP. [DOI](https://doi.org/10.1109/TIP.2003.819861)。
- [scikit-image 指标定义](https://scikit-image.org/docs/stable/api/skimage.metrics.html)、[PyTorch 可复现性](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。
'''
    (ROOT/'report.md').write_text(report)
    print('report.md')


if __name__=='__main__': main()
