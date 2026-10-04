# Ubuntu 运行说明

## 安装

```bash
git clone https://github.com/guanqingbao/Bioimage-pannel.git
cd Bioimage-pannel
chmod +x scripts/*.sh
./scripts/setup_ubuntu.sh
```

## 启动

```bash
./scripts/start_ubuntu.sh
```

服务默认监听 `127.0.0.1:8010`，处理数据保存在项目的 `data/` 目录。

如果 PDF 和处理结果放在其他磁盘，启动前设置：

```bash
export IMAGE_BATCH_PDF_ROOTS=/data/pdfs
export IMAGE_BATCH_DATA_DIR=/你的数据目录/image-batch
./scripts/start_ubuntu.sh
```

`IMAGE_BATCH_PDF_ROOTS` 是前端目录选择器可以访问的边界。多个根目录使用冒号分隔，例如
`/data/pdfs:/mnt/archive/pdfs`；前端不能浏览或处理这些根目录之外的路径。若不设置，
启动脚本会创建并使用项目中的 `pdfs/`。

从 Windows 访问服务器时，先建立 SSH 隧道：

```powershell
ssh -L 8010:127.0.0.1:8010 yh@服务器IP
```

然后打开 <http://127.0.0.1:8010>。

在页面点击“选择目录”即可浏览上述 PDF 根目录；勾选“包含子文件夹”后，处理结果会按照原始相对目录在左侧导航中分组。这台 32 物理核服务器建议先使用并发 16，稳定后再测试 24。

## 验证

```bash
curl http://127.0.0.1:8010/api/health
.venv/bin/python -m unittest discover -s tests -v
```
