# python100

## 激光角点提取软件 (Laser Corner Extraction System)

### 运行环境 / Requirements

安装依赖：

```bash
pip install -r requirements.txt
```

### 启动程序 / Run

```bash
python laser_corner_extraction.py
```

### 登录账号 / Login Credentials

程序启动后会弹出登录窗口，使用以下演示账号登录：

| 用户名 / Username | 密码 / Password |
|:-----------------:|:---------------:|
| admin             | admin123        |
| user              | user123         |

> 提示：以上账号信息也显示在登录窗口底部。

### 中文显示 / Chinese Text Rendering

图表中的中文标签需要系统安装 CJK 字体才能正常显示，否则会出现乱码（方块）。

| 操作系统 | 状态 |
|----------|------|
| Windows  | ✅ 已内置 SimHei / Microsoft YaHei，开箱即用 |
| macOS    | ✅ 已内置 PingFang SC / Heiti SC，开箱即用 |
| Linux    | ⚠️ 需手动安装字体，执行以下命令后重启程序 |

```bash
# Linux (Debian / Ubuntu)
sudo apt-get install -y fonts-wqy-microhei

# Linux (Fedora / RHEL / CentOS)
sudo dnf install -y wqy-microhei-fonts

# Linux (Arch)
sudo pacman -S --noconfirm wqy-microhei
```

