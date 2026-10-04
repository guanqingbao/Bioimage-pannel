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

如果希望把数据放在其他磁盘，启动前设置：

```bash
export IMAGE_BATCH_DATA_DIR=/你的数据目录/image-batch
./scripts/start_ubuntu.sh
```

从 Windows 访问服务器时，先建立 SSH 隧道：

```powershell
ssh -L 8010:127.0.0.1:8010 yh@服务器IP
```

然后打开 <http://127.0.0.1:8010>。

服务器文件夹批处理需要填写 Ubuntu 路径，例如 `/data/pdfs`。这台 32 物理核服务器建议先使用并发 16，稳定后再测试 24。

## 验证

```bash
curl http://127.0.0.1:8010/api/health
.venv/bin/python -m unittest discover -s tests -v
```
