# 数据来源与本地预览

## 来源与使用条件

来源：[DIV2K 官方页面](https://data.vision.ee.ethz.ch/cvl/DIV2K/)，核对日期 2026-10-12。

官方说明数据供学术研究使用，图片版权归原作者。项目仅将其用作研究中的场景纹理代理，不声称照片由本项目拍摄或属于公有领域。原始照片保存在 `data/`，照片衍生的演示图片与数组保存在 `results/data_preview/`；这两个目录均被 Git 忽略。仓库可保留获取程序、参数、划分与来源记录。公开展示照片或其衍生图前，另行核对相应使用权限；自生成图案可继续用作公开演示。

按官方要求引用数据集论文与挑战报告：

- Eirikur Agustsson and Radu Timofte. *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.* CVPR Workshops, 2017.
- Radu Timofte et al. *NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results.* CVPR Workshops, 2017.

本项目不是超分辨率挑战复现，也不使用官方低分辨率图片作为观测。我们从 HR 图片生成自己的线性灰度场景，然后按光学模型合成观测。

## 当前取得的数据

最初从官方 `DIV2K_train_HR.zip` 按 HTTP Range 获取两张训练候选照片：0356、0223，用于教学。随后取得清单中的全部 150 张，原图总大小约 671.1 MB，没有下载完整的约 3.53 GB 数据包。ZIP 成员在读取时检查 CRC；原图 SHA-256 与服务器 ETag 另有记录。

`results/data_preparation/candidate_split.json` 记录了按种子 20261012 从官方 800 张训练原图中生成的项目候选划分：100 张训练、20 张验证、30 张测试。它是项目自定义划分，不是官方挑战划分。

全部原图已预处理为 384×384 线性灰度场景，并完成以下排查：原文件、解码像素和预处理像素的精确哈希；完整图与中心场景的感知哈希（包含 90° 旋转和镜像）；助手查看全部来源缩略图。精确哈希与感知哈希阈值检查没有标出重复对。目视发现 0207 与 0655 是同一博物馆展品的不同视角，两张均在训练集内，记录为相关来源组。

未发现明显的跨组重复照片；哈希和缩略图检查仍可能漏掉部分同场景照片，不能证明绝对独立。没有基于测试集复原表现调整任何照片。数据划分已保存为 `results/data_preparation/data_manifest.json`，核查记录为 `duplicate_screening.json` 与 `duplicate_review.json`。目前仅冻结来源划分；模型结构、训练预算与完整正式实验配置仍待训练前确定。

## 固定的预览处理

1. 按 EXIF 修正方向；有 ICC 时转为 sRGB，无 ICC 时明确按 sRGB 假设处理。
2. 先做 sRGB 逆传递，再按 0.2126 R + 0.7152 G + 0.0722 B 计算线性灰度。
3. 用 BOX 面积平均缩小至短边 384 像素，居中裁出 384×384 场景；不逐图拉伸对比度。
4. 在完整场景上做周期卷积：失焦 a=0.15 waves，光瞳直径 128，FFT 网格 512，核宽 129；加 σ=0.01 的高斯噪声。
5. 各图噪声种子为 100000 + 原图整数 ID；Wiener 演示用 K=0.001，未通过验证集选优。
6. 各面板共用显示范围 [0,1]，保存的浮点数组不截断。局部图统一取场景中央 128×128，不依据结果选择位置。

线性灰度按同一范围直接显示，外观可能比原始照片更暗。这不是曝光实验，也不是测量真实相机；相机成像、后期处理与预缩放都已存在于场景代理的构建中。全图边缘可能出现周期边界伪影。Wiener 验证选参在运行前固定使用中央 256×256 区域；早期教学局部图仍保持中央 128×128。中央裁剪仅减弱边界影响，不保证消除反卷积的所有边界效应。

## 复现当前样本

```bash
.venv/bin/python scripts/fetch_div2k_preview.py
.venv/bin/python prepare_data_preview.py
```

第一条命令需要联网；第二条使用本地文件。网络读取若不支持 HTTP Range 会停止，不自动下载整个数据包。

预览：`results/data_preview/natural_samples.png`；局部图：`0356_detail.png`、`0223_detail.png`。来源哈希、处理方法与参数位于 `results/data_preparation/preview_acquisition.json` 和 `preview_processing.json`。

这批输出仅用于检查数据流程与学习，没有 CNN 输出，没有正式测试成绩。

## 用户参与的 K 调整

用户观察到 Wiener 结果“噪点变多，特别是天空”，并选择增大 K 以减轻噪点。助手把该方向落实为从 0.001 到 0.01 的教学比较：

```bash
.venv/bin/python compare_photo_wiener.py --k 0.01
```

程序复用保存的观测与 PSF，不重新抽取噪声；先核对原 K 的重算结果与缓存一致，再生成两个 K 的对比。原始预览文件保留，输出为 `results/data_preview/0356_k_comparison.png` 与 `0223_k_comparison.png`。L1 沿用同一中央区域和统一截断规则，记录在 `results/data_preparation/wiener_k_comparison.json`。这一步不选择正式最佳 K。

## 完整数据流程

```bash
.venv/bin/python scripts/fetch_dataset.py
.venv/bin/python scripts/prepare_dataset.py
.venv/bin/python scripts/freeze_data_split.py
.venv/bin/python scripts/select_wiener.py
```

前两条完成获取、预处理及重复筛查。第三条要求人工复核记录 `duplicate_review.json` 与当前数据指纹一致；不能将自动筛查等同于目视检查。第四条仅在冻结的验证集上选 K，细节见 `STEP_04_VALIDATION.md`。重跑时如来源、划分或核查结果发生变化，应重新审查并记录原因，程序不会静默覆盖不同的已冻结划分。
