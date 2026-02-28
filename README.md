# python100

## 激光角点提取系统 (Laser Corner Extraction System)

一个基于 OpenCV 的激光图像角点提取 GUI 软件，具备登录/退出功能。

### 功能特性

- **用户登录 / 退出** – 用户名密码验证，支持多账户
- **图像加载** – 支持 PNG / JPG / BMP / TIFF 等主流格式
- **角点检测算法**
  - Harris 角点检测（可调块大小、K 值、阈值）
  - Shi-Tomasi 角点检测（可调最大角点数、质量等级、最小距离）
  - FAST 角点检测（可调阈值、非极大值抑制开关）
- **Canny 边缘检测**（可调低/高阈值）
- **图像直方图** – RGB 三通道分布可视化
- **结果保存** – 将检测结果导出为图像文件
- **快捷键** – Ctrl+O 打开、Ctrl+S 保存、Ctrl+Q 退出

### 安装与运行

```bash
pip install -r requirements.txt
python laser_corner_extraction.py
```

### 默认账户

| 用户名  | 密码       |
|---------|------------|
| admin   | admin123   |
| user    | user123    |

### 依赖

- Python 3.8+
- opencv-python
- numpy
- Pillow
- matplotlib