# Bioimage-pannel（Image Batch）

面向大批量科研 PDF 的本地视觉数据采集项目。它只处理图片，不执行论文文本抽取或大模型语义分析：

1. 从 PDF 中定位完整的 `Figure` / `Scheme`；
2. 统一按 **300 DPI** 渲染整图；
3. 使用本地 OpenCV/版面规则自动划分 A–Z 子图；
4. 在浏览器中人工新增或修正框选，并保留每一次图片版本；
5. 用 SQLite 保存批次和单篇任务状态，适合持续积累大规模图像数据。

服务意外关闭后再次启动时，会自动把中断的单篇任务放回队列并继续执行。

项目复用了当前仓库中已经回归验证过的 `pdf_figures` 和 `panel_splitter`，运行时不需要 API Key，也不会调用视觉大模型。

## 目录结构

```text
Bioimage-pannel/
├── app.py                 # FastAPI 服务入口
├── image_batch/           # PDF 整图提取、子图划分和训练数据整理
├── static/                # 浏览器前端
├── scripts/               # Windows / Ubuntu 安装与启动脚本
├── docs/                  # 部署说明
├── tests/                 # 回归测试
├── requirements.txt       # Python 依赖
└── README.md
```

运行数据默认写入 `data/`，该目录不会提交到 Git，首次启动时会自动创建。

## 启动

推荐创建独立环境：

```powershell
cd Bioimage-pannel
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\start_windows.ps1
```

打开 <http://127.0.0.1:8010>。

Ubuntu 服务器部署请阅读 [`docs/ubuntu.md`](docs/ubuntu.md)。

如果已经安装了仓库根目录的依赖，也可以直接运行：

```powershell
cd Bioimage-pannel
python -m uvicorn app:app --host 127.0.0.1 --port 8010
```

## 两种批量入口

- **浏览器多选上传**：适合几十到几百篇；单文件上限默认 80 MB，单批最多 2000 篇。
- **服务器文件夹**：直接填写本机 PDF 文件夹，可递归扫描；适合万篇到十万篇，不会先复制一份原 PDF。

并发数默认按 CPU 核数计算，最大不超过 32。建议先用 100～1000 篇分别测试并发 4、8、16，观察 CPU、内存、磁盘吞吐和失败率后再扩大。

## 输出目录

```text
data/
├── batches.sqlite3                 # 批次队列与进度
└── <job_id>/
    ├── upload/                     # 浏览器上传的原 PDF；文件夹模式不复制原文
    ├── job_result.json             # 单篇结果入口
    ├── editor_state.json           # 前端编辑状态
    └── results/
        ├── extracted/<paper>/
        │   ├── figures/            # 300 DPI 整图
        │   └── figures_metadata.json
        └── documents/<paper>/panels/<figure>/
            ├── layout.json         # 自动切分坐标与诊断
            ├── preview.png         # 划分预览
            ├── panel_A.png         # 当前子图
            └── manual_versions/    # 人工框选的不可变历史版本
```

人工修正不会覆盖历史：自动版本保留为 `v0000_automatic.png`，之后依次保存为 `v0001_manual.png`、`v0002_manual.png`。

## 未划分训练素材库

前端中的自动结果先作为 `proposed` 候选。人工可以新增、重画或删除面板；确认所有面板框后点击“确认面板划分完成”。如果整张 Figure 本身不需要划分，则点击“确认整图无需划分”，系统会保存 `annotation_mode: no_split`，并生成合法的空 YOLO 标签（不会伪造整图框）。只有 `verified + annotation_complete` 样本才进入可训练集合。

点击页面顶部“整理训练数据”，会生成：

```text
data/training_collection/
├── assets/images/                 # 按 SHA-256 去重的完整 Figure
├── annotations/<sample_id>.json   # 所有 proposed/verified/ambiguous 统一标注
├── ready/
│   ├── images/<sample_id>.png     # 已完整审核、尚未划分数据集的训练图
│   └── labels/<sample_id>.txt     # 单类别 YOLO 标签：0 main_panel
├── manifests/images.jsonl         # 全部样本清单
├── manifests/ready.jsonl          # 可进入后续划分的白名单
└── collection_summary.json        # 数量、状态和失败记录
```

这里暂不划分 `train/val/test`。后续划分时应按 `source_group_id`（PDF 文件哈希）分组，避免同一论文的 Figure 跨集合泄漏。

## API

- `POST /api/batches/upload`：多文件上传并创建批次。
- `POST /api/batches/folder`：扫描服务器文件夹并创建批次。
- `GET /api/batches`：最近批次。
- `GET /api/batches/{batch_id}`：分页查看批次进度。
- `GET /api/jobs/{job_id}/result`：读取单篇结果。
- `POST /api/jobs/{job_id}/figures/{record_id}/panel-versions`：保存人工裁框版本。
- `DELETE /api/jobs/{job_id}/figures/{record_id}/panels`：停用错误框，历史仍保留。
- `POST /api/jobs/{job_id}/figures/{record_id}/review`：保存整图审核状态。
- `POST /api/training-collection/rebuild`：重建未划分训练素材库与清单。
