#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光角点提取软件 (Laser Corner Point Extraction Software)
功能: 激光扫描数据的角点提取、可视化与分析
Features: Corner extraction, visualization and analysis for laser scan data
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# Support Chinese characters in matplotlib plots (cross-platform font fallbacks)
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei',               # Windows (built-in)
    'Microsoft YaHei',      # Windows (built-in)
    'PingFang SC',          # macOS (built-in)
    'Heiti SC',             # macOS (built-in)
    'STHeiti',              # macOS older
    'WenQuanYi Micro Hei',  # Linux (apt: fonts-wqy-microhei)
    'Noto Sans CJK SC',     # Linux / macOS (apt: fonts-noto-cjk)
    'Source Han Sans CN',
    'DejaVu Sans',          # fallback (no CJK, but avoids crash)
]
matplotlib.rcParams['axes.unicode_minus'] = False  # prevent minus sign garbling
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import json
import hashlib
import csv
from datetime import datetime
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────── user store (demo) ──────────────────────────────
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


USERS = {
    "admin":    _hash("admin123"),
    "user":     _hash("user123"),
}


# ═══════════════════════════════════════════════════════════════════════════
#                           LOGIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════
class LoginWindow:
    """Login dialog – must succeed before the main window opens."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.authenticated_user: str | None = None

        self.root.title("激光角点提取软件 – 登录")
        self.root.resizable(False, False)
        self._center(320, 260)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── layout ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = tk.Frame(self.root, bg="#2C3E50", padx=20, pady=20)
        outer.pack(fill=tk.BOTH, expand=True)

        tk.Label(outer, text="🔬 激光角点提取软件",
                 font=("Microsoft YaHei", 14, "bold"),
                 bg="#2C3E50", fg="white").pack(pady=(0, 4))
        tk.Label(outer, text="Laser Corner Extraction System",
                 font=("Arial", 9), bg="#2C3E50", fg="#BDC3C7").pack(pady=(0, 16))

        form = tk.Frame(outer, bg="#2C3E50")
        form.pack(fill=tk.X)

        tk.Label(form, text="用户名 / Username:", bg="#2C3E50",
                 fg="white", anchor="w").grid(row=0, column=0, sticky="w", pady=4)
        self._user_var = tk.StringVar(value="admin")
        ttk.Entry(form, textvariable=self._user_var, width=22).grid(
            row=0, column=1, pady=4, padx=(8, 0))

        tk.Label(form, text="密  码 / Password:", bg="#2C3E50",
                 fg="white", anchor="w").grid(row=1, column=0, sticky="w", pady=4)
        self._pass_var = tk.StringVar()
        self._pass_entry = ttk.Entry(form, textvariable=self._pass_var,
                                     show="*", width=22)
        self._pass_entry.grid(row=1, column=1, pady=4, padx=(8, 0))
        self._pass_entry.bind("<Return>", lambda _: self._login())

        self._msg_var = tk.StringVar()
        tk.Label(outer, textvariable=self._msg_var, bg="#2C3E50",
                 fg="#E74C3C", font=("Arial", 9)).pack(pady=4)

        btn_frame = tk.Frame(outer, bg="#2C3E50")
        btn_frame.pack()
        ttk.Button(btn_frame, text="登录  Login",
                   command=self._login).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="退出  Exit",
                   command=self._on_close).pack(side=tk.LEFT, padx=6)

        tk.Label(outer, text="演示账号: admin / admin123  |  user / user123",
                 bg="#2C3E50", fg="#F0E68C", font=("Arial", 9, "bold")).pack(pady=(12, 0))

    def _login(self):
        username = self._user_var.get().strip()
        password = self._pass_var.get()
        if username in USERS and USERS[username] == _hash(password):
            self.authenticated_user = username
            self.root.destroy()
        else:
            self._msg_var.set("用户名或密码错误！")
            self._pass_var.set("")

    def _on_close(self):
        self.root.destroy()

    def _center(self, w: int, h: int):
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")


# ═══════════════════════════════════════════════════════════════════════════
#                        CORNER-EXTRACTION ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════
class LaserCornerExtractor:
    """Algorithms for 2-D laser-scan corner extraction."""

    # ── public entry ────────────────────────────────────────────────────────
    @staticmethod
    def extract(points: np.ndarray,
                method: str = "curvature",
                sigma: float = 1.5,
                threshold: float = 0.05,
                min_distance: int = 5) -> np.ndarray:
        """
        points : (N,2) array of (x,y)
        Returns (M,2) array of corner points.
        """
        if len(points) < 10:
            return np.empty((0, 2))
        x, y = points[:, 0], points[:, 1]
        if method == "curvature":
            idx = LaserCornerExtractor._curvature(x, y, sigma, threshold, min_distance)
        elif method == "harris":
            idx = LaserCornerExtractor._harris(x, y, sigma, threshold, min_distance)
        elif method == "bearing_angle":
            idx = LaserCornerExtractor._bearing_angle(x, y, threshold, min_distance)
        elif method == "line_segment":
            idx = LaserCornerExtractor._line_segment(x, y, threshold, min_distance)
        else:
            idx = np.array([], dtype=int)
        return points[idx]

    # ── method 1: curvature ─────────────────────────────────────────────────
    @staticmethod
    def _curvature(x, y, sigma, threshold, min_dist):
        xs = gaussian_filter1d(x.astype(float), sigma)
        ys = gaussian_filter1d(y.astype(float), sigma)
        dx  = np.gradient(xs);  dy  = np.gradient(ys)
        ddx = np.gradient(dx);  ddy = np.gradient(dy)
        denom = (dx**2 + dy**2)**1.5
        with np.errstate(divide='ignore', invalid='ignore'):
            kappa = np.abs(dx*ddy - dy*ddx) / np.where(denom < 1e-10, 1e-10, denom)
        kappa = gaussian_filter1d(kappa, sigma)
        peak_thresh = threshold * kappa.max() if kappa.max() > 0 else threshold
        return LaserCornerExtractor._find_peaks(kappa, peak_thresh, min_dist)

    # ── method 2: Harris ────────────────────────────────────────────────────
    @staticmethod
    def _harris(x, y, sigma, threshold, min_dist):
        xs = gaussian_filter1d(x.astype(float), sigma)
        ys = gaussian_filter1d(y.astype(float), sigma)
        dx = np.gradient(xs);  dy = np.gradient(ys)
        Ixx = gaussian_filter1d(dx**2,    sigma)
        Iyy = gaussian_filter1d(dy**2,    sigma)
        Ixy = gaussian_filter1d(dx * dy,  sigma)
        k = 0.04
        det   = Ixx*Iyy - Ixy**2
        trace = Ixx + Iyy
        R = det - k * trace**2
        thresh = threshold * R.max() if R.max() > 0 else threshold
        return LaserCornerExtractor._find_peaks(np.maximum(R, 0), thresh, min_dist)

    # ── method 3: Bearing-Angle ─────────────────────────────────────────────
    @staticmethod
    def _bearing_angle(x, y, threshold, min_dist):
        angles = np.zeros(len(x))
        for i in range(1, len(x) - 1):
            v1 = np.array([x[i-1]-x[i], y[i-1]-y[i]])
            v2 = np.array([x[i+1]-x[i], y[i+1]-y[i]])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(v1, v2)/(n1*n2), -1, 1)
                angles[i] = np.degrees(np.arccos(cos_a))
        thresh = threshold * 180.0          # threshold as fraction of 180°
        return LaserCornerExtractor._find_peaks(angles, thresh, min_dist)

    # ── method 4: Line-Segment intersection ─────────────────────────────────
    @staticmethod
    def _line_segment(x, y, threshold, min_dist):
        """Split into line segments via split-and-merge; return endpoints."""
        n = len(x)
        if n < 6:
            return np.array([], dtype=int)
        segments = LaserCornerExtractor._recursive_split(
            x, y, 0, n-1, threshold * 0.5)
        seg_endpoints = set()
        for s, e in segments:
            seg_endpoints.add(s)
            seg_endpoints.add(e)
        # also mark interior boundaries
        boundaries = sorted(seg_endpoints)
        # skip first and last (they are scan endpoints)
        return np.array([b for b in boundaries[1:-1]], dtype=int)

    @staticmethod
    def _recursive_split(x, y, start, end, tol):
        if end - start < 3:
            return [(start, end)]
        # distance of each point from the line start→end
        dx, dy = x[end]-x[start], y[end]-y[start]
        L = np.hypot(dx, dy)
        if L < 1e-10:
            dists = np.hypot(x[start:end+1]-x[start], y[start:end+1]-y[start])
        else:
            dists = np.abs(dy*x[start:end+1] - dx*y[start:end+1]
                           + x[end]*y[start] - y[end]*x[start]) / L
        max_idx = np.argmax(dists) + start
        if dists[max_idx - start] > tol:
            left  = LaserCornerExtractor._recursive_split(x, y, start,   max_idx, tol)
            right = LaserCornerExtractor._recursive_split(x, y, max_idx, end,     tol)
            return left + right
        return [(start, end)]

    # ── shared peak-picking ──────────────────────────────────────────────────
    @staticmethod
    def _find_peaks(signal, threshold, min_dist):
        indices = []
        i = 0
        n = len(signal)
        while i < n:
            if signal[i] >= threshold:
                # collect the run above threshold
                j = i
                while j < n and signal[j] >= threshold:
                    j += 1
                peak = i + np.argmax(signal[i:j])
                if not indices or (peak - indices[-1]) >= min_dist:
                    indices.append(peak)
                i = j
            else:
                i += 1
        return np.array(indices, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════
#                           MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════
class MainApp:
    """Main application window."""

    METHODS = {
        "曲率法 (Curvature)":          "curvature",
        "Harris 角点检测":              "harris",
        "方位角法 (Bearing-Angle)":     "bearing_angle",
        "线段分割法 (Line-Segment)":    "line_segment",
    }

    _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, root: tk.Tk, username: str):
        self.root = root
        self.username = username
        self.points: np.ndarray | None = None
        self.corners: np.ndarray | None = None
        self.filepath: str = ""
        self._bg_image: np.ndarray | None = None

        self.root.title("🔬 激光角点提取软件  –  Laser Corner Extraction System")
        # Maximize window (cross-platform)
        try:
            self.root.state("zoomed")        # Windows
        except tk.TclError:
            try:
                self.root.attributes("-zoomed", True)   # Linux/X11
            except tk.TclError:
                self.root.geometry("1280x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._log(f"欢迎, {username}！  系统就绪。")

    # ── top-level layout ────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_menu()
        self._build_toolbar()

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        left  = self._build_left_panel(paned)
        center = self._build_center_panel(paned)
        right  = self._build_right_panel(paned)

        paned.add(left,   weight=0)
        paned.add(center, weight=3)
        paned.add(right,  weight=1)

        self._build_status_bar()

    # ── menu ────────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = tk.Menu(self.root)
        self.root.config(menu=mb)

        # File
        fm = tk.Menu(mb, tearoff=False)
        mb.add_cascade(label="文件 File", menu=fm)
        fm.add_command(label="打开数据文件… Open…",
                       accelerator="Ctrl+O", command=self._open_file)
        fm.add_command(label="生成演示数据  Generate Sample",
                       command=self._generate_sample)
        fm.add_separator()
        fm.add_command(label="导出角点 CSV  Export CSV",
                       accelerator="Ctrl+S", command=self._export_csv)
        fm.add_command(label="导出角点 JSON  Export JSON",
                       command=self._export_json)
        fm.add_separator()
        fm.add_command(label="退出登录  Logout", command=self._logout)
        fm.add_command(label="退出程序  Exit",
                       accelerator="Alt+F4",  command=self._on_close)

        # Edit / Settings
        em = tk.Menu(mb, tearoff=False)
        mb.add_cascade(label="设置 Settings", menu=em)
        em.add_command(label="重置参数  Reset Parameters", command=self._reset_params)
        em.add_command(label="清除结果  Clear Results",   command=self._clear_results)

        # View
        vm = tk.Menu(mb, tearoff=False)
        mb.add_cascade(label="视图 View", menu=vm)
        vm.add_command(label="显示原始数据  Show Raw Data",    command=lambda: self._toggle_layer("raw"))
        vm.add_command(label="显示角点      Show Corners",     command=lambda: self._toggle_layer("corners"))
        vm.add_command(label="显示连线      Show Lines",       command=lambda: self._toggle_layer("lines"))
        vm.add_separator()
        vm.add_command(label="重置视图  Reset View", command=self._reset_view)

        # Help
        hm = tk.Menu(mb, tearoff=False)
        mb.add_cascade(label="帮助 Help", menu=hm)
        hm.add_command(label="使用说明  User Guide", command=self._show_guide)
        hm.add_command(label="关于  About",           command=self._show_about)

        self.root.bind("<Control-o>", lambda _: self._open_file())
        self.root.bind("<Control-s>", lambda _: self._export_csv())

    # ── toolbar ─────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        tb = ttk.Frame(self.root, relief=tk.RIDGE, padding=2)
        tb.pack(fill=tk.X, padx=0, pady=0)

        btns = [
            ("📂 打开",    self._open_file),
            ("🧪 演示数据", self._generate_sample),
            ("⚙️ 提取角点", self._run_extraction),
            ("💾 导出CSV",  self._export_csv),
            ("🔄 重置",     self._reset_view),
            ("🚪 退出登录", self._logout),
        ]
        for label, cmd in btns:
            ttk.Button(tb, text=label, command=cmd, width=11).pack(
                side=tk.LEFT, padx=2, pady=1)

        # spacer + user badge
        ttk.Label(tb, text="").pack(side=tk.LEFT, expand=True)
        ttk.Label(tb, text=f"👤 {self.username}",
                  font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=8)

    # ── left panel ──────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        frame = ttk.Frame(parent, width=210)
        frame.pack_propagate(False)

        # ── file info ──
        fi = ttk.LabelFrame(frame, text="文件信息 File Info", padding=6)
        fi.pack(fill=tk.X, padx=4, pady=4)
        self._file_label  = ttk.Label(fi, text="未加载  (no file)",
                                      wraplength=180, foreground="gray")
        self._file_label.pack(anchor="w")
        self._points_label = ttk.Label(fi, text="点数: --")
        self._points_label.pack(anchor="w")

        # ── algorithm ──
        alg = ttk.LabelFrame(frame, text="算法 Algorithm", padding=6)
        alg.pack(fill=tk.X, padx=4, pady=4)
        self._method_var = tk.StringVar(value=list(self.METHODS.keys())[0])
        for name in self.METHODS:
            ttk.Radiobutton(alg, text=name, variable=self._method_var,
                            value=name).pack(anchor="w")

        # ── parameters ──
        pm = ttk.LabelFrame(frame, text="参数 Parameters", padding=6)
        pm.pack(fill=tk.X, padx=4, pady=4)

        params = [
            ("平滑σ (Sigma):",     "sigma",    "1.5"),
            ("阈值 (Threshold):",  "threshold","0.05"),
            ("最小间距 (Min Dist):","min_dist", "5"),
        ]
        self._param_vars: dict[str, tk.StringVar] = {}
        for label, key, default in params:
            ttk.Label(pm, text=label).pack(anchor="w")
            var = tk.StringVar(value=default)
            self._param_vars[key] = var
            ttk.Entry(pm, textvariable=var, width=12).pack(anchor="w", pady=1)

        # ── preprocess ──
        pre = ttk.LabelFrame(frame, text="预处理 Preprocessing", padding=6)
        pre.pack(fill=tk.X, padx=4, pady=4)
        self._smooth_var  = tk.BooleanVar(value=True)
        self._outlier_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pre, text="高斯平滑  Gaussian smooth",
                        variable=self._smooth_var).pack(anchor="w")
        ttk.Checkbutton(pre, text="去除离群点  Remove outliers",
                        variable=self._outlier_var).pack(anchor="w")

        # ── run button ──
        ttk.Button(frame, text="▶  执行提取  Run Extraction",
                   command=self._run_extraction).pack(fill=tk.X, padx=4, pady=8)

        return frame

    # ── center panel ────────────────────────────────────────────────────────
    def _build_center_panel(self, parent):
        frame = ttk.Frame(parent)

        self._fig = Figure(figsize=(8, 6), dpi=96, facecolor="#FAFAFA")
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_title("激光扫描点云  /  Laser Scan Point Cloud",
                           fontsize=11)
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.grid(True, alpha=0.3)
        self._ax.set_aspect("equal")

        self._canvas = FigureCanvasTkAgg(self._fig, master=frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nav = NavigationToolbar2Tk(self._canvas, frame)
        nav.update()

        return frame

    # ── right panel ─────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        frame = ttk.Frame(parent, width=220)
        frame.pack_propagate(False)

        # corners list
        cl = ttk.LabelFrame(frame, text="检测角点  Detected Corners", padding=4)
        cl.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._corner_count = ttk.Label(cl, text="共 0 个角点")
        self._corner_count.pack(anchor="w")

        cols = ("No.", "X (m)", "Y (m)")
        self._tree = ttk.Treeview(cl, columns=cols, show="headings", height=16)
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=60, anchor="center")
        vsb = ttk.Scrollbar(cl, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # statistics
        st = ttk.LabelFrame(frame, text="统计 Statistics", padding=6)
        st.pack(fill=tk.X, padx=4, pady=4)
        self._stats_var = tk.StringVar(value="尚未提取  (not extracted)")
        ttk.Label(st, textvariable=self._stats_var,
                  wraplength=200, justify="left").pack(anchor="w")

        # log
        lg = ttk.LabelFrame(frame, text="日志 Log", padding=4)
        lg.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._log_text = tk.Text(lg, height=8, state=tk.DISABLED,
                                 font=("Courier", 8), wrap=tk.WORD)
        sb = ttk.Scrollbar(lg, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        return frame

    # ── status bar ──────────────────────────────────────────────────────────
    def _build_status_bar(self):
        bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding=2)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_var = tk.StringVar(value="就绪  Ready")
        ttk.Label(bar, textvariable=self._status_var, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        self._time_label = ttk.Label(bar, text="")
        self._time_label.pack(side=tk.RIGHT, padx=8)
        self._update_clock()

    def _update_clock(self):
        self._time_label.config(
            text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._update_clock)

    # ══════════════════════════════════════════════════════════════════════
    #                            ACTIONS
    # ══════════════════════════════════════════════════════════════════════

    # ── open file ───────────────────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="打开数据/图片文件  Open data/image file",
            filetypes=[("支持的文件", "*.txt *.csv *.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("文本/CSV文件", "*.txt *.csv"),
                       ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("所有文件", "*.*")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in self._IMAGE_EXTS:
            self._open_image(path)
            return
        try:
            data = []
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    nums = []
                    for cell in row:
                        try:
                            nums.append(float(cell.strip()))
                        except ValueError:
                            pass
                    if len(nums) >= 2:
                        data.append(nums[:2])
                    elif len(nums) == 1:
                        # polar single-column (just range) – skip header
                        pass
            if not data:
                messagebox.showerror("错误", "文件中未找到有效数值列对。\n"
                                     "请确保文件每行含 X Y 两列数值。\n\n"
                                     "如需处理图片文件（JPG/PNG），请在打开对话框中\n"
                                     "选择'图片文件'类型。\n"
                                     "To load images, select 'Image files' in the dialog.")
                return
            pts = np.array(data, dtype=float)
            # auto-detect polar (angle_deg, range) vs cartesian (x, y)
            if np.all(pts[:, 0] >= -360) and np.all(pts[:, 0] <= 360):
                col2_max = pts[:, 1].max()
                col1_range = pts[:, 0].max() - pts[:, 0].min()
                if col1_range > 10 and col2_max < 200:   # likely polar
                    ang = np.radians(pts[:, 0])
                    r   = pts[:, 1]
                    pts = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
            self.points   = pts
            self.filepath = path
            self._bg_image = None
            self._file_label.config(
                text=os.path.basename(path), foreground="black")
            self._points_label.config(text=f"点数: {len(pts)}")
            self._clear_results()
            self._draw()
            self._log(f"已加载文件: {os.path.basename(path)}  ({len(pts)} 点)")
            self._status("文件已加载  File loaded")
        except Exception as exc:
            messagebox.showerror("读取错误", str(exc))

    # ── open image file ─────────────────────────────────────────────────────
    def _open_image(self, path: str):
        """Load an image file and extract edge points as a 2-D point cloud."""
        try:
            import matplotlib.image as mpimg
            from scipy.ndimage import sobel

            img = mpimg.imread(path)
            # Convert to grayscale float in [0, 1]
            if img.ndim == 3:
                gray = np.mean(img[:, :, :3], axis=2)
            else:
                gray = img.astype(float)
            if gray.max() > 1:
                gray = gray / 255.0

            # Edge detection via Sobel magnitude
            sx = sobel(gray, axis=0)
            sy = sobel(gray, axis=1)
            edge_mag = np.hypot(sx, sy)

            thresh = np.mean(edge_mag) + np.std(edge_mag)
            rows, cols = np.where(edge_mag > thresh)

            if len(cols) == 0:
                messagebox.showerror(
                    "错误", "未能从图片中检测到有效边缘点。\n"
                    "No valid edge points detected from the image.")
                return

            # Use image coordinates: x = col, y = row (top-down)
            pts = np.column_stack([cols.astype(float), rows.astype(float)])

            # Sub-sample when there are too many edge pixels
            max_pts = 10000
            if len(pts) > max_pts:
                idx = np.random.default_rng(0).choice(
                    len(pts), max_pts, replace=False)
                idx.sort()
                pts = pts[idx]

            self.points = pts
            self.filepath = path
            self._bg_image = gray
            self._file_label.config(
                text=os.path.basename(path), foreground="black")
            self._points_label.config(text=f"点数: {len(pts)}")
            self._clear_results()
            self._draw()
            self._log(f"已加载图片: {os.path.basename(path)}  "
                       f"({len(pts)} 边缘点)")
            self._status("图片已加载  Image loaded")
        except Exception as exc:
            messagebox.showerror("读取错误", str(exc))

    # ── generate sample data ────────────────────────────────────────────────
    def _generate_sample(self):
        """Generate a synthetic indoor-like laser scan (room outline)."""
        rng = np.random.default_rng(42)
        # Room outline: four walls forming a rectangle, plus some obstacles
        angles = np.linspace(-np.pi * 0.85, np.pi * 0.85, 720)
        ranges = np.zeros(len(angles))

        # wall distances as function of angle (axis-aligned rectangle 8×6 m)
        for i, a in enumerate(angles):
            ca, sa = np.cos(a), np.sin(a)
            # intersection with box ±4 x ±3
            tx = (4.0 / abs(ca)) if abs(ca) > 1e-6 else 1e9
            ty = (3.0 / abs(sa)) if abs(sa) > 1e-6 else 1e9
            r  = min(tx, ty)
            # add a box obstacle at (2, 1) size 1×1
            # (simplified: bump in range)
            ranges[i] = r

        # obstacle: pillar at (2.5, 1.0) radius 0.3
        for i, a in enumerate(angles):
            ca, sa = np.cos(a), np.sin(a)
            # ray vs circle: (ca*t-2.5)^2 + (sa*t-1.0)^2 = 0.09
            A = 1.0
            B = -2*(ca*2.5 + sa*1.0)
            C = 2.5**2 + 1.0**2 - 0.09
            disc = B**2 - 4*A*C
            if disc >= 0:
                t = (-B - np.sqrt(disc)) / (2*A)
                if 0 < t < ranges[i]:
                    ranges[i] = t

        # add Gaussian noise
        ranges += rng.normal(0, 0.015, len(ranges))
        ranges = np.clip(ranges, 0.1, 20)

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        self.points = np.column_stack([x, y])
        self.filepath = ""
        self._bg_image = None
        self._file_label.config(text="演示数据 (sample)", foreground="#27AE60")
        self._points_label.config(text=f"点数: {len(self.points)}")
        self._clear_results()
        self._draw()
        self._log("已生成演示激光扫描数据（720 点）")
        self._status("演示数据已生成  Sample data generated")

    # ── run extraction ──────────────────────────────────────────────────────
    def _run_extraction(self):
        if self.points is None:
            messagebox.showwarning("提示", "请先加载或生成数据！")
            return
        try:
            sigma     = float(self._param_vars["sigma"].get())
            threshold = float(self._param_vars["threshold"].get())
            min_dist  = int(self._param_vars["min_dist"].get())
        except ValueError:
            messagebox.showerror("参数错误", "参数格式不正确，请输入数字。")
            return

        pts = self.points.copy()
        if self._smooth_var.get():
            from scipy.ndimage import gaussian_filter1d as gf
            pts = np.column_stack([gf(pts[:, 0], sigma),
                                   gf(pts[:, 1], sigma)])
        if self._outlier_var.get():
            pts = self._remove_outliers(pts)

        method_key = self._method_var.get()
        method     = self.METHODS[method_key]
        self._status(f"正在提取角点…  Extracting with {method}")
        self.root.update_idletasks()

        self.corners = LaserCornerExtractor.extract(
            pts, method=method,
            sigma=sigma, threshold=threshold, min_distance=min_dist)

        self._update_corner_list()
        self._draw()
        n = len(self.corners)
        self._log(f"提取完成: 方法={method_key}, 角点数={n}, "
                  f"σ={sigma}, 阈值={threshold}, 最小间距={min_dist}")
        self._status(f"提取完成  Done – {n} corners detected")

    # ── outlier removal ──────────────────────────────────────────────────────
    @staticmethod
    def _remove_outliers(pts: np.ndarray, z: float = 3.0) -> np.ndarray:
        if len(pts) < 4:
            return pts
        d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        mean, std = d.mean(), d.std()
        keep = np.concatenate([[True],
                               np.abs(d - mean) < z * std])
        return pts[keep]

    # ── drawing ─────────────────────────────────────────────────────────────
    def _draw(self):
        self._ax.clear()
        if self._bg_image is not None:
            h, w = self._bg_image.shape[:2]
            self._ax.imshow(self._bg_image, cmap='gray', alpha=0.5,
                            extent=[0, w, h, 0], aspect='equal')
            self._ax.set_title("图片边缘点云  /  Image Edge Points",
                               fontsize=11)
            self._ax.set_xlabel("X (px)")
            self._ax.set_ylabel("Y (px)")
        else:
            self._ax.set_title("激光扫描点云  /  Laser Scan Point Cloud",
                               fontsize=11)
            self._ax.set_xlabel("X (m)")
            self._ax.set_ylabel("Y (m)")
        self._ax.grid(True, alpha=0.3)
        self._ax.set_aspect("equal")

        if self.points is not None:
            self._ax.plot(self.points[:, 0], self.points[:, 1],
                          "b.", ms=1.5, alpha=0.6, label="扫描点 Scan")

        if self.corners is not None and len(self.corners):
            self._ax.scatter(self.corners[:, 0], self.corners[:, 1],
                             c="red", s=60, zorder=5, marker="*",
                             edgecolors="darkred", linewidths=0.5,
                             label=f"角点 Corners ({len(self.corners)})")
            # annotate indices
            for i, (cx, cy) in enumerate(self.corners):
                self._ax.annotate(str(i+1), (cx, cy),
                                  textcoords="offset points",
                                  xytext=(4, 4), fontsize=7, color="darkred")

        if self.points is not None:
            self._ax.legend(loc="upper right", fontsize=8)

        self._canvas.draw_idle()

    def _toggle_layer(self, layer):
        # Simple toggle – just redraw (layers always shown for now)
        self._draw()

    def _reset_view(self):
        self._ax.autoscale()
        self._canvas.draw_idle()

    # ── corner list ──────────────────────────────────────────────────────────
    def _update_corner_list(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        if self.corners is None or len(self.corners) == 0:
            self._corner_count.config(text="共 0 个角点")
            self._stats_var.set("未检测到角点")
            return
        for i, (cx, cy) in enumerate(self.corners):
            self._tree.insert("", tk.END, values=(i+1, f"{cx:.4f}", f"{cy:.4f}"))
        n = len(self.corners)
        self._corner_count.config(text=f"共 {n} 个角点")
        xs, ys = self.corners[:, 0], self.corners[:, 1]
        stats = (f"数量: {n}\n"
                 f"X 范围: [{xs.min():.3f}, {xs.max():.3f}]\n"
                 f"Y 范围: [{ys.min():.3f}, {ys.max():.3f}]\n"
                 f"重心: ({xs.mean():.3f}, {ys.mean():.3f})")
        self._stats_var.set(stats)

    # ── export ──────────────────────────────────────────────────────────────
    def _export_csv(self):
        if self.corners is None or len(self.corners) == 0:
            messagebox.showwarning("提示", "没有角点数据可导出，请先执行提取。")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            title="保存角点CSV  Save corners CSV")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["No.", "X (m)", "Y (m)"])
            for i, (cx, cy) in enumerate(self.corners):
                w.writerow([i+1, round(cx, 6), round(cy, 6)])
        self._log(f"角点已导出至: {os.path.basename(path)}")
        self._status(f"CSV 已保存  Saved: {os.path.basename(path)}")
        messagebox.showinfo("导出成功", f"角点数据已保存至:\n{path}")

    def _export_json(self):
        if self.corners is None or len(self.corners) == 0:
            messagebox.showwarning("提示", "没有角点数据可导出，请先执行提取。")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            title="保存角点JSON  Save corners JSON")
        if not path:
            return
        payload = {
            "software":  "激光角点提取软件",
            "timestamp": datetime.now().isoformat(),
            "method":    self._method_var.get(),
            "count":     len(self.corners),
            "corners": [{"id": i+1, "x": round(float(cx), 6), "y": round(float(cy), 6)}
                        for i, (cx, cy) in enumerate(self.corners)],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._log(f"角点已导出 JSON: {os.path.basename(path)}")
        messagebox.showinfo("导出成功", f"角点 JSON 已保存至:\n{path}")

    # ── misc actions ────────────────────────────────────────────────────────
    def _reset_params(self):
        defaults = {"sigma": "1.5", "threshold": "0.05", "min_dist": "5"}
        for k, v in defaults.items():
            self._param_vars[k].set(v)
        self._log("参数已重置  Parameters reset")

    def _clear_results(self):
        self.corners = None
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._corner_count.config(text="共 0 个角点")
        self._stats_var.set("尚未提取  (not extracted)")
        self._draw()

    def _logout(self):
        if messagebox.askyesno("退出登录", "确定要退出登录？\nConfirm logout?"):
            self.root.destroy()
            _launch_login()

    def _on_close(self):
        if messagebox.askyesno("退出程序", "确定退出程序？\nExit application?"):
            self.root.destroy()

    def _show_about(self):
        messagebox.showinfo(
            "关于  About",
            "🔬 激光角点提取软件\n"
            "Laser Corner Point Extraction System\n\n"
            "版本 Version: 1.0.0\n"
            "算法: 曲率法 / Harris / 方位角法 / 线段分割法\n"
            "Algorithms: Curvature / Harris / Bearing-Angle / Line-Segment\n\n"
            "支持格式: TXT / CSV / JPG / PNG / BMP / TIFF\n"
            "Export: CSV / JSON")

    def _show_guide(self):
        win = tk.Toplevel(self.root)
        win.title("使用说明  User Guide")
        win.geometry("520x420")
        text = tk.Text(win, wrap=tk.WORD, font=("Arial", 10), padx=12, pady=8)
        text.pack(fill=tk.BOTH, expand=True)
        guide = """激光角点提取软件 使用说明
═══════════════════════════════════════

【数据格式 / Data Format】
• TXT / CSV 文件，每行两列数值：
  - 笛卡尔坐标: X(m)  Y(m)
  - 极坐标: Angle(deg)  Range(m)
• 图片文件 (JPG / PNG / BMP / TIFF)：
  - 自动提取图片边缘点作为二维点云

【操作步骤 / Steps】
1. 点击"打开"加载数据文件或图片，或点击"演示数据"生成测试数据。
2. 在左侧选择角点提取算法：
   • 曲率法 (Curvature): 适合一般场景
   • Harris 角点检测: 经典角点算法
   • 方位角法 (Bearing-Angle): 适合稀疏点云
   • 线段分割法 (Line-Segment): 适合室内规则场景
3. 调整参数：
   • 平滑σ: 高斯平滑强度 (推荐 1~3)
   • 阈值:  角点判定阈值 (推荐 0.02~0.1)
   • 最小间距: 角点最近相邻点数 (推荐 3~10)
4. 点击"执行提取"查看结果。
5. 在右侧面板查看检测到的角点坐标。
6. 点击"导出CSV"或菜单"导出JSON"保存结果。

【算法说明 / Algorithm Notes】
• 曲率法: 计算点序列曲率，提取高曲率点
• Harris: 构造二维结构张量矩阵，计算角点响应
• 方位角法: 计算相邻向量夹角，阈值判断角点
• 线段分割: 递归分割法拟合线段，取断点为角点
"""
        text.insert(tk.END, guide)
        text.config(state=tk.DISABLED)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _status(self, msg: str):
        self._status_var.set(msg)
        self.root.update_idletasks()


# ═══════════════════════════════════════════════════════════════════════════
#                               ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def _launch_login():
    login_root = tk.Tk()
    login = LoginWindow(login_root)
    login_root.mainloop()
    user = login.authenticated_user
    if user:
        main_root = tk.Tk()
        MainApp(main_root, user)
        main_root.mainloop()


if __name__ == "__main__":
    _launch_login()
