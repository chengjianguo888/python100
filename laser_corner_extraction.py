"""
激光角点提取系统  v2.0
Laser Corner Extraction System

一个基于 OpenCV 的激光图像角点提取 GUI 软件，支持多种角点检测与边缘检测算法。
v2.0 新增：高斯预处理、CLAHE 增强、亚像素精化、ORB 检测、热力图叠加、CSV 导出。

使用方法:
    pip install -r requirements.txt
    python laser_corner_extraction.py

默认账户:
    用户名: admin  密码: admin123
    用户名: user   密码: user123
"""

import csv
import hashlib
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


def _hash_password(password: str) -> str:
    """Return the SHA-256 hex digest of *password*."""
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_corner(img: np.ndarray, x: int, y: int, color, r: int = 8) -> None:
    """Draw a crosshair + ring marker for a single corner point."""
    cv2.circle(img, (x, y), r, color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)
    arm = r + 4
    cv2.line(img, (x - arm, y), (x - r + 1, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x + r - 1, y), (x + arm, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - arm), (x, y - r + 1), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y + r - 1), (x, y + arm), color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# User credentials: values are SHA-256 hashes of the passwords.
# (In production use a proper auth backend with bcrypt/argon2 + salts.)
# ---------------------------------------------------------------------------
USERS = {
    "admin": _hash_password("admin123"),
    "user":  _hash_password("user123"),
}

# Sub-pixel refinement criteria used by Harris and Shi-Tomasi
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
_SUBPIX_WIN = (5, 5)


# ---------------------------------------------------------------------------
# Login window
# ---------------------------------------------------------------------------

class LoginWindow:
    """Login screen shown before the main application window."""

    def __init__(self, root: tk.Tk, on_success):
        self.root = root
        self.on_success = on_success

        self.root.title("激光角点提取系统 – 登录")
        self.root.resizable(False, False)
        self._center(400, 320)
        self._build_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _center(self, w: int, h: int) -> None:
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _build_ui(self) -> None:
        # Header
        hdr = tk.Frame(self.root, bg="#2c3e50", height=80)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr,
            text="激光角点提取系统",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#2c3e50",
        ).pack(pady=20)

        # Form
        form = tk.Frame(self.root, pady=10)
        form.pack(padx=50, fill=tk.BOTH, expand=True)

        tk.Label(form, text="用户名:", font=("Arial", 11)).grid(
            row=0, column=0, sticky="w", pady=8
        )
        self._username = tk.StringVar()
        tk.Entry(form, textvariable=self._username, font=("Arial", 11), width=24).grid(
            row=0, column=1, pady=8, padx=6
        )

        tk.Label(form, text="密  码:", font=("Arial", 11)).grid(
            row=1, column=0, sticky="w", pady=8
        )
        self._password = tk.StringVar()
        tk.Entry(
            form,
            textvariable=self._password,
            font=("Arial", 11),
            width=24,
            show="*",
        ).grid(row=1, column=1, pady=8, padx=6)

        tk.Button(
            form,
            text="登  录",
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            width=22,
            pady=5,
            command=self._login,
        ).grid(row=2, column=0, columnspan=2, pady=15)

        self._status = tk.StringVar()
        tk.Label(
            self.root, textvariable=self._status, fg="red", font=("Arial", 9)
        ).pack()

        # Hint
        tk.Label(
            self.root,
            text="默认账户  admin / admin123",
            fg="#7f8c8d",
            font=("Arial", 8),
        ).pack(pady=(0, 8))

        self.root.bind("<Return>", lambda _e: self._login())

        # Focus username field
        form.grid_slaves(row=0, column=1)[0].focus_set()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _login(self) -> None:
        username = self._username.get().strip()
        password = self._password.get()
        if not username or not password:
            self._status.set("请输入用户名和密码")
            return
        if USERS.get(username) == _hash_password(password):
            self.on_success(username)
        else:
            self._status.set("用户名或密码错误，请重试")
            self._password.set("")


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class LaserCornerExtractionApp:
    """Main GUI application for laser corner extraction."""

    def __init__(self, root: tk.Tk, username: str, on_logout):
        self.root = root
        self.username = username
        self.on_logout = on_logout

        self._src_img = None       # original BGR image (numpy array)
        self._result_img = None    # processed BGR image (numpy array)
        self._corners = []         # list of (x, y) tuples
        self._corner_responses = []  # response strengths (float) per corner
        self._tk_img_orig = None   # PhotoImage reference (prevents GC)
        self._tk_img_proc = None

        self.root.title(f"激光角点提取系统  –  当前用户: {username}")
        self.root.geometry("1360x860")

        self._build_menu()
        self._build_ui()
        self._status("欢迎使用激光角点提取系统！请点击「打开图像」开始。")

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_menu(self) -> None:
        mb = tk.Menu(self.root)
        self.root.config(menu=mb)

        # File
        fm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="文件", menu=fm)
        fm.add_command(label="打开图像  Ctrl+O", command=self._open_image)
        fm.add_command(label="保存结果  Ctrl+S", command=self._save_result)
        fm.add_separator()
        fm.add_command(label="退出登录", command=self._logout)
        fm.add_command(label="退出程序  Ctrl+Q", command=self._quit)

        # Process
        pm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="处理", menu=pm)
        pm.add_command(label="Harris 角点检测", command=lambda: self._detect("Harris"))
        pm.add_command(label="Shi-Tomasi 角点检测", command=lambda: self._detect("Shi-Tomasi"))
        pm.add_command(label="FAST 角点检测", command=lambda: self._detect("FAST"))
        pm.add_command(label="ORB 特征检测", command=lambda: self._detect("ORB"))
        pm.add_command(label="激光亮斑角点", command=lambda: self._detect("激光亮斑"))
        pm.add_separator()
        pm.add_command(label="Canny 边缘检测", command=self._canny)
        pm.add_command(label="清除结果", command=self._clear)

        # View
        vm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="视图", menu=vm)
        vm.add_command(label="图像直方图", command=self._histogram)
        vm.add_command(label="导出角点坐标 CSV", command=self._export_csv)
        vm.add_command(label="重置视图", command=self._reset_view)

        # Help
        hm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="帮助", menu=hm)
        hm.add_command(label="使用说明", command=self._help)
        hm.add_command(label="关于", command=self._about)

        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda _e: self._open_image())
        self.root.bind("<Control-s>", lambda _e: self._save_result())
        self.root.bind("<Control-q>", lambda _e: self._quit())

    def _build_ui(self) -> None:
        # ── Outer frame ──────────────────────────────────────────────
        outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Left control panel ────────────────────────────────────────
        left = tk.Frame(outer, width=310, bg="#ecf0f1", bd=1, relief=tk.SUNKEN)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
        left.pack_propagate(False)

        tk.Label(
            left,
            text="控 制 面 板",
            font=("Arial", 13, "bold"),
            bg="#2c3e50",
            fg="white",
        ).pack(fill=tk.X)

        # — Load section —
        lf = tk.LabelFrame(left, text="图像加载", font=("Arial", 10), bg="#ecf0f1", padx=5, pady=5)
        lf.pack(fill=tk.X, padx=6, pady=6)
        tk.Button(lf, text="打开图像", command=self._open_image,
                  font=("Arial", 10), bg="#3498db", fg="white", width=22).pack(pady=3)
        self._info_var = tk.StringVar(value="尚未加载图像")
        tk.Label(lf, textvariable=self._info_var, bg="#ecf0f1",
                 font=("Arial", 9), wraplength=250, justify=tk.LEFT).pack()

        # — Preprocessing section —
        pref = tk.LabelFrame(left, text="预处理 (激光增强)", font=("Arial", 10), bg="#ecf0f1", padx=5, pady=5)
        pref.pack(fill=tk.X, padx=6, pady=3)

        self._p_blur = tk.BooleanVar(value=True)
        blur_row = tk.Frame(pref, bg="#ecf0f1")
        blur_row.pack(fill=tk.X)
        tk.Checkbutton(blur_row, text="高斯模糊", variable=self._p_blur,
                       bg="#ecf0f1", font=("Arial", 9)).pack(side=tk.LEFT)
        tk.Label(blur_row, text="核:", bg="#ecf0f1", font=("Arial", 9)).pack(side=tk.LEFT)
        self._p_blur_k = tk.IntVar(value=3)
        tk.Spinbox(blur_row, from_=1, to=15, increment=2, textvariable=self._p_blur_k,
                   width=5, font=("Arial", 9)).pack(side=tk.LEFT, padx=2)

        self._p_clahe = tk.BooleanVar(value=False)
        clahe_row = tk.Frame(pref, bg="#ecf0f1")
        clahe_row.pack(fill=tk.X)
        tk.Checkbutton(clahe_row, text="CLAHE 对比度增强", variable=self._p_clahe,
                       bg="#ecf0f1", font=("Arial", 9)).pack(side=tk.LEFT)

        self._p_subpix = tk.BooleanVar(value=True)
        tk.Checkbutton(pref, text="亚像素精化 (Harris/Shi-T)", variable=self._p_subpix,
                       bg="#ecf0f1", font=("Arial", 9)).pack(anchor="w")

        # — Algorithm selector —
        af = tk.LabelFrame(left, text="角点检测", font=("Arial", 10), bg="#ecf0f1", padx=5, pady=5)
        af.pack(fill=tk.X, padx=6, pady=6)

        tk.Label(af, text="检测算法:", bg="#ecf0f1", font=("Arial", 9)).pack(anchor="w")
        self._algo_var = tk.StringVar(value="Harris")
        algo_cb = ttk.Combobox(
            af,
            textvariable=self._algo_var,
            values=["Harris", "Shi-Tomasi", "FAST", "ORB", "激光亮斑"],
            state="readonly",
            width=24,
        )
        algo_cb.pack(pady=(3, 5))
        algo_cb.bind("<<ComboboxSelected>>", self._on_algo_change)

        # Harris params
        self._pf_harris = tk.LabelFrame(af, text="Harris 参数", bg="#ecf0f1", font=("Arial", 9), padx=4, pady=4)
        _row = 0
        for label, var_name, default, dtype in [
            ("块大小:", "_p_block",  2,    int),
            ("K 值:",   "_p_k",     0.04, float),
            ("阈  值:", "_p_thresh", 0.01, float),
        ]:
            tk.Label(self._pf_harris, text=label, bg="#ecf0f1", font=("Arial", 9)).grid(
                row=_row, column=0, sticky="w", pady=2
            )
            v = tk.DoubleVar(value=default) if dtype == float else tk.IntVar(value=default)
            setattr(self, var_name, v)
            tk.Entry(self._pf_harris, textvariable=v, width=12).grid(
                row=_row, column=1, padx=4
            )
            _row += 1
        # Aperture must be one of 1, 3, 5, 7 — use Combobox to avoid invalid values
        tk.Label(self._pf_harris, text="光圈大小:", bg="#ecf0f1", font=("Arial", 9)).grid(
            row=_row, column=0, sticky="w", pady=2
        )
        self._p_aperture = tk.IntVar(value=3)
        ttk.Combobox(
            self._pf_harris, textvariable=self._p_aperture,
            values=[1, 3, 5, 7], state="readonly", width=9,
        ).grid(row=_row, column=1, padx=4)

        # Shi-Tomasi params
        self._pf_shi = tk.LabelFrame(af, text="Shi-Tomasi 参数", bg="#ecf0f1", font=("Arial", 9), padx=4, pady=4)
        for _r, (label, var_name, default, dtype) in enumerate([
            ("最大角点数:", "_p_max_corners",  100,  int),
            ("质量等级:",   "_p_quality",      0.01, float),
            ("最小距离:",   "_p_min_dist",     10,   int),
        ]):
            tk.Label(self._pf_shi, text=label, bg="#ecf0f1", font=("Arial", 9)).grid(
                row=_r, column=0, sticky="w", pady=2
            )
            v = tk.DoubleVar(value=default) if dtype == float else tk.IntVar(value=default)
            setattr(self, var_name, v)
            tk.Entry(self._pf_shi, textvariable=v, width=12).grid(row=_r, column=1, padx=4)

        # FAST params
        self._pf_fast = tk.LabelFrame(af, text="FAST 参数", bg="#ecf0f1", font=("Arial", 9), padx=4, pady=4)
        tk.Label(self._pf_fast, text="阈  值:", bg="#ecf0f1", font=("Arial", 9)).grid(
            row=0, column=0, sticky="w", pady=2
        )
        self._p_fast_thresh = tk.IntVar(value=10)
        tk.Entry(self._pf_fast, textvariable=self._p_fast_thresh, width=12).grid(
            row=0, column=1, padx=4
        )
        self._p_nonmax = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self._pf_fast, text="非极大值抑制", variable=self._p_nonmax,
            bg="#ecf0f1", font=("Arial", 9),
        ).grid(row=1, column=0, columnspan=2, sticky="w")

        # ORB params
        self._pf_orb = tk.LabelFrame(af, text="ORB 参数", bg="#ecf0f1", font=("Arial", 9), padx=4, pady=4)
        for _r, (label, var_name, default, dtype) in enumerate([
            ("最大特征数:", "_p_orb_n",     500,  int),
            ("层数:",       "_p_orb_levels", 8,    int),
        ]):
            tk.Label(self._pf_orb, text=label, bg="#ecf0f1", font=("Arial", 9)).grid(
                row=_r, column=0, sticky="w", pady=2
            )
            v = tk.IntVar(value=default)
            setattr(self, var_name, v)
            tk.Entry(self._pf_orb, textvariable=v, width=12).grid(row=_r, column=1, padx=4)

        # 激光亮斑 params
        self._pf_laser = tk.LabelFrame(af, text="激光亮斑参数", bg="#ecf0f1", font=("Arial", 9), padx=4, pady=4)
        for _r, (label, var_name, default, dtype) in enumerate([
            ("亮度阈值:", "_p_laser_bright", 180, int),
            ("最小面积:", "_p_laser_minarea", 5,   int),
        ]):
            tk.Label(self._pf_laser, text=label, bg="#ecf0f1", font=("Arial", 9)).grid(
                row=_r, column=0, sticky="w", pady=2
            )
            v = tk.IntVar(value=default)
            setattr(self, var_name, v)
            tk.Entry(self._pf_laser, textvariable=v, width=12).grid(row=_r, column=1, padx=4)

        self._on_algo_change(None)   # show Harris panel by default

        tk.Button(
            af, text="执行角点检测",
            command=self._run_detection,
            font=("Arial", 10, "bold"),
            bg="#27ae60", fg="white", width=22,
        ).pack(pady=(8, 3))

        # — Canny section —
        cf = tk.LabelFrame(left, text="边缘检测 (Canny)", font=("Arial", 10), bg="#ecf0f1", padx=5, pady=5)
        cf.pack(fill=tk.X, padx=6, pady=6)
        for _r, (lbl, var_name, default) in enumerate([
            ("低阈值:", "_p_canny_low",  50),
            ("高阈值:", "_p_canny_high", 150),
        ]):
            tk.Label(cf, text=lbl, bg="#ecf0f1", font=("Arial", 9)).grid(row=_r, column=0, sticky="w", pady=2)
            v = tk.IntVar(value=default)
            setattr(self, var_name, v)
            tk.Spinbox(cf, from_=0, to=255, textvariable=v, width=10).grid(row=_r, column=1, padx=4)
        tk.Button(cf, text="边缘检测", command=self._canny,
                  font=("Arial", 10), bg="#e67e22", fg="white", width=22).grid(
                  row=2, column=0, columnspan=2, pady=(8, 3))

        # — Results section —
        rf = tk.LabelFrame(left, text="检测结果", font=("Arial", 10), bg="#ecf0f1", padx=5, pady=5)
        rf.pack(fill=tk.X, padx=6, pady=6)
        self._count_var = tk.StringVar(value="角点数量: 0")
        tk.Label(rf, textvariable=self._count_var, bg="#ecf0f1",
                 font=("Arial", 11, "bold")).pack()
        tk.Button(rf, text="保存结果图像", command=self._save_result,
                  font=("Arial", 10), bg="#9b59b6", fg="white", width=22).pack(pady=3)
        tk.Button(rf, text="导出角点 CSV", command=self._export_csv,
                  font=("Arial", 10), bg="#16a085", fg="white", width=22).pack(pady=3)
        tk.Button(rf, text="清除结果", command=self._clear,
                  font=("Arial", 10), bg="#95a5a6", fg="white", width=22).pack(pady=3)

        # — Logout at bottom —
        bot = tk.Frame(left, bg="#ecf0f1")
        bot.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        tk.Label(bot, text=f"登录用户: {self.username}", bg="#ecf0f1",
                 font=("Arial", 9, "italic")).pack(anchor="w")
        tk.Button(bot, text="退出登录", command=self._logout,
                  font=("Arial", 10, "bold"), bg="#e74c3c", fg="white", width=22).pack(pady=4)

        # ── Right display area ────────────────────────────────────────
        right = tk.Frame(outer, bg="#2c3e50")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._nb = ttk.Notebook(right)
        self._nb.pack(fill=tk.BOTH, expand=True)

        orig_f = tk.Frame(self._nb, bg="#1a252f")
        self._nb.add(orig_f, text="  原始图像  ")
        self._canvas_orig = tk.Canvas(orig_f, bg="#1a252f", cursor="crosshair")
        self._canvas_orig.pack(fill=tk.BOTH, expand=True)
        self._canvas_orig.bind("<Motion>", self._on_mouse)

        proc_f = tk.Frame(self._nb, bg="#1a252f")
        self._nb.add(proc_f, text="  处理结果  ")
        self._canvas_proc = tk.Canvas(proc_f, bg="#1a252f", cursor="crosshair")
        self._canvas_proc.pack(fill=tk.BOTH, expand=True)

        # ── Status bar ────────────────────────────────────────────────
        sb = tk.Frame(self.root, bg="#2c3e50", height=26)
        sb.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_var = tk.StringVar()
        tk.Label(sb, textvariable=self._status_var, fg="white",
                 bg="#2c3e50", font=("Arial", 9), anchor="w").pack(side=tk.LEFT, padx=10)
        self._coord_var = tk.StringVar()
        tk.Label(sb, textvariable=self._coord_var, fg="white",
                 bg="#2c3e50", font=("Arial", 9), anchor="e").pack(side=tk.RIGHT, padx=10)

    # ==================================================================
    # Helper utilities
    # ==================================================================

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)

    def _show_result(self, img: np.ndarray) -> None:
        """Switch to the result tab and render *img* on the processed canvas."""
        self._nb.select(1)
        self.root.update_idletasks()
        self._show_image(img, self._canvas_proc, "_tk_img_proc")

    def _on_algo_change(self, _event) -> None:
        for frame in (self._pf_harris, self._pf_shi, self._pf_fast, self._pf_orb, self._pf_laser):
            frame.pack_forget()
        algo = self._algo_var.get()
        if algo == "Harris":
            self._pf_harris.pack(fill=tk.X, pady=4)
        elif algo == "Shi-Tomasi":
            self._pf_shi.pack(fill=tk.X, pady=4)
        elif algo == "FAST":
            self._pf_fast.pack(fill=tk.X, pady=4)
        elif algo == "ORB":
            self._pf_orb.pack(fill=tk.X, pady=4)
        elif algo == "激光亮斑":
            self._pf_laser.pack(fill=tk.X, pady=4)

    def _on_mouse(self, event) -> None:
        if self._src_img is not None:
            self._coord_var.set(f"坐标: ({event.x}, {event.y})")

    def _show_image(self, img: np.ndarray, canvas: tk.Canvas, store_attr: str) -> None:
        """Fit *img* (BGR or GRAY) into *canvas* and keep a PhotoImage reference."""
        canvas.delete("all")
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        # winfo_width/height returns 1 for hidden/un-rendered widgets — use safe fallback
        if cw < 50:
            cw = 900
        if ch < 50:
            ch = 650

        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        setattr(self, store_attr, photo)   # prevent garbage collection
        x0 = (cw - nw) // 2
        y0 = (ch - nh) // 2
        canvas.create_image(x0, y0, anchor=tk.NW, image=photo)

    # ==================================================================
    # File operations
    # ==================================================================

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("错误", "无法读取图像，请检查文件格式。")
            return
        self._src_img = img
        self._result_img = None
        self._corners = []
        self._count_var.set("角点数量: 0")
        h, w = img.shape[:2]
        self._info_var.set(f"文件: {os.path.basename(path)}\n尺寸: {w} × {h} 像素")
        self._show_image(img, self._canvas_orig, "_tk_img_orig")
        self._canvas_proc.delete("all")
        self._nb.select(0)
        self._status(f"已加载: {os.path.basename(path)}  ({w}×{h})")

    def _save_result(self) -> None:
        if self._result_img is None:
            messagebox.showwarning("提示", "暂无处理结果可保存，请先执行检测。")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果图像",
            defaultextension=".png",
            filetypes=[
                ("PNG 文件", "*.png"),
                ("JPEG 文件", "*.jpg"),
                ("BMP 文件", "*.bmp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        cv2.imwrite(path, self._result_img)
        self._status(f"结果已保存: {path}")
        messagebox.showinfo("保存成功", f"结果图像已保存至:\n{path}")

    def _export_csv(self) -> None:
        if not self._corners:
            messagebox.showwarning("提示", "暂无角点数据，请先执行检测。")
            return
        path = filedialog.asksaveasfilename(
            title="导出角点坐标",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "response"])
            for i, ((x, y), r) in enumerate(
                zip(self._corners, self._corner_responses or [0.0] * len(self._corners))
            ):
                writer.writerow([i + 1, x, y, f"{r:.6f}"])
        self._status(f"已导出 {len(self._corners)} 个角点坐标: {path}")
        messagebox.showinfo("导出成功", f"角点坐标已保存至:\n{path}")

    # ==================================================================
    # Detection algorithms
    # ==================================================================

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """Apply laser-image preprocessing pipeline to a grayscale image."""
        out = gray.copy()
        if self._p_blur.get():
            k = int(self._p_blur_k.get())
            k = k if k % 2 == 1 else k + 1   # kernel must be odd
            out = cv2.GaussianBlur(out, (k, k), 0)
        if self._p_clahe.get():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            out = clahe.apply(out)
        return out

    def _run_detection(self) -> None:
        self._detect(self._algo_var.get())

    def _detect(self, algorithm: str) -> None:
        if self._src_img is None:
            messagebox.showwarning("提示", "请先打开一张图像。")
            return
        try:
            gray = cv2.cvtColor(self._src_img, cv2.COLOR_BGR2GRAY)
            proc = self._preprocess(gray)   # preprocessed grayscale
            result = self._src_img.copy()
            corners = []
            responses = []

            if algorithm == "Harris":
                block_size = max(1, int(self._p_block.get()))
                # Aperture is selected from Combobox — must be in {1, 3, 5, 7}
                _valid_apertures = (1, 3, 5, 7)
                aperture_raw = int(self._p_aperture.get())
                aperture = min(_valid_apertures, key=lambda a: abs(a - aperture_raw))
                k = float(self._p_k.get())
                thresh = float(self._p_thresh.get())

                gray_f = np.float32(proc)
                dst = cv2.cornerHarris(gray_f, block_size, aperture, k)
                dst = cv2.dilate(dst, None)

                # Blend response heat-map onto result for visual context
                dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                heat = cv2.applyColorMap(dst_norm, cv2.COLORMAP_JET)
                result = cv2.addWeighted(result, 0.75, heat, 0.25, 0)

                # Connected-component NMS → proper corner centres
                binary = (dst > thresh * dst.max()).astype(np.uint8)
                _, _, _, centroids = cv2.connectedComponentsWithStats(binary)
                centroid_pts = np.float32([[cx, cy] for cx, cy in centroids[1:]])   # skip background

                # Sub-pixel refinement
                if len(centroid_pts) > 0 and self._p_subpix.get():
                    refined = cv2.cornerSubPix(
                        proc, centroid_pts.reshape(-1, 1, 2), _SUBPIX_WIN, (-1, -1), _SUBPIX_CRITERIA
                    )
                    centroid_pts = refined.reshape(-1, 2)

                h_dst, w_dst = dst.shape
                for cx, cy in centroid_pts:
                    x, y = int(round(float(cx))), int(round(float(cy)))
                    # Clamp to valid image coordinates before response lookup
                    xi, yi = max(0, min(x, w_dst - 1)), max(0, min(y, h_dst - 1))
                    resp = float(dst[yi, xi])
                    corners.append((x, y))
                    responses.append(resp)
                    _draw_corner(result, x, y, (0, 255, 255))   # cyan crosshair

            elif algorithm == "Shi-Tomasi":
                max_c = int(self._p_max_corners.get())
                quality = float(self._p_quality.get())
                min_d = int(self._p_min_dist.get())
                pts = cv2.goodFeaturesToTrack(proc, max_c, quality, min_d)
                if pts is not None:
                    # Sub-pixel refinement
                    if self._p_subpix.get():
                        pts = cv2.cornerSubPix(proc, pts, _SUBPIX_WIN, (-1, -1), _SUBPIX_CRITERIA)
                    for pt in pts:
                        x, y = int(round(pt[0][0])), int(round(pt[0][1]))
                        corners.append((x, y))
                        responses.append(1.0)
                        _draw_corner(result, x, y, (0, 255, 0))   # green crosshair

            elif algorithm == "FAST":
                fast = cv2.FastFeatureDetector_create(
                    threshold=int(self._p_fast_thresh.get()),
                    nonmaxSuppression=bool(self._p_nonmax.get()),
                )
                kps = fast.detect(proc, None)
                # Sort by response strength descending
                kps = sorted(kps, key=lambda kp: kp.response, reverse=True)
                max_resp = kps[0].response if kps else 1.0
                for kp in kps:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    # Scale marker radius by relative response
                    r = max(4, int(6 * kp.response / max_resp) + 4)
                    _draw_corner(result, x, y, (255, 100, 0), r=r)
                    corners.append((x, y))
                    responses.append(float(kp.response))

            elif algorithm == "ORB":
                n = int(self._p_orb_n.get())
                levels = int(self._p_orb_levels.get())
                orb = cv2.ORB_create(nfeatures=n, nlevels=levels)
                kps = orb.detect(proc, None)
                kps = sorted(kps, key=lambda kp: kp.response, reverse=True)
                max_resp = kps[0].response if kps else 1.0
                for kp in kps:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    r = max(4, int(8 * kp.response / max_resp) + 3)
                    _draw_corner(result, x, y, (255, 0, 200), r=r)
                    corners.append((x, y))
                    responses.append(float(kp.response))

            elif algorithm == "激光亮斑":
                # Isolate bright laser spots via threshold + morphology → contour corners
                bright_thresh = int(self._p_laser_bright.get())
                min_area = int(self._p_laser_minarea.get())
                _, binary = cv2.threshold(proc, bright_thresh, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Show threshold overlay in blue channel
                overlay = result.copy()
                overlay[binary > 0] = (overlay[binary > 0] * 0.5 + np.array([80, 0, 0])).clip(0, 255).astype(np.uint8)
                result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area:
                        continue
                    # Sub-pixel centroid
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    x, y = int(round(cx)), int(round(cy))
                    r = max(4, int(np.sqrt(area / np.pi)))
                    _draw_corner(result, x, y, (0, 200, 255), r=r)
                    corners.append((x, y))
                    responses.append(float(area))

            self._result_img = result
            self._corners = corners
            self._corner_responses = responses
            self._show_result(result)
            self._count_var.set(f"角点数量: {len(corners)}")
            self._status(f"{algorithm} 检测完成 – 检测到 {len(corners)} 个角点")

        except Exception as exc:
            messagebox.showerror("错误", f"角点检测失败:\n{exc}")

    def _canny(self) -> None:
        if self._src_img is None:
            messagebox.showwarning("提示", "请先打开一张图像。")
            return
        try:
            gray = cv2.cvtColor(self._src_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, int(self._p_canny_low.get()), int(self._p_canny_high.get()))
            self._result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self._show_result(edges)
            count = int(np.count_nonzero(edges))
            self._status(f"Canny 边缘检测完成 – 边缘像素数: {count}")
        except Exception as exc:
            messagebox.showerror("错误", f"边缘检测失败:\n{exc}")

    # ==================================================================
    # View operations
    # ==================================================================

    def _histogram(self) -> None:
        if self._src_img is None:
            messagebox.showwarning("提示", "请先打开一张图像。")
            return
        win = tk.Toplevel(self.root)
        win.title("图像 RGB 直方图")
        win.geometry("640x420")
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (color, name) in enumerate(zip(("b", "g", "r"), ("Blue", "Green", "Red"))):
            hist = cv2.calcHist([self._src_img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=name, alpha=0.75)
        ax.set_xlabel("像素值")
        ax.set_ylabel("频率")
        ax.set_title("RGB 直方图")
        ax.legend()
        ax.grid(True, alpha=0.3)
        cvs = FigureCanvasTkAgg(fig, master=win)
        cvs.draw()
        cvs.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def _reset_view(self) -> None:
        if self._src_img is not None:
            self._show_image(self._src_img, self._canvas_orig, "_tk_img_orig")
        if self._result_img is not None:
            self._show_image(self._result_img, self._canvas_proc, "_tk_img_proc")

    def _clear(self) -> None:
        self._result_img = None
        self._corners = []
        self._canvas_proc.delete("all")
        self._count_var.set("角点数量: 0")
        self._status("已清除检测结果")

    # ==================================================================
    # Help / About
    # ==================================================================

    def _help(self) -> None:
        messagebox.showinfo(
            "使用说明",
            "激光角点提取系统 v2.0 – 使用说明\n\n"
            "1. 打开图像：点击「打开图像」或使用 Ctrl+O\n"
            "2. 预处理：开启高斯模糊 / CLAHE 增强（推荐激光图像）\n"
            "3. 选择算法：从下拉菜单选择角点检测算法\n"
            "4. 调整参数：根据需要修改算法参数\n"
            "5. 执行检测：点击「执行角点检测」按钮\n"
            "6. 查看结果：在「处理结果」标签页查看\n"
            "7. 导出结果：保存图像或导出角点坐标 CSV\n\n"
            "支持的算法：\n"
            "• Harris       – 经典角点检测，含热力图叠加\n"
            "• Shi-Tomasi   – 改进的 Harris，精度更高\n"
            "• FAST         – 高速角点检测\n"
            "• ORB          – 鲁棒特征检测算法\n"
            "• 激光亮斑     – 针对激光点/线的专用检测\n"
            "• Canny        – 边缘提取\n\n"
            "v2.0 新功能：\n"
            "  • 亚像素精化 (Harris / Shi-Tomasi)\n"
            "  • 高斯模糊 + CLAHE 预处理\n"
            "  • Harris 热力图叠加显示\n"
            "  • CSV 角点坐标导出\n"
            "  • ORB 与激光亮斑新算法\n\n"
            "快捷键：\n"
            "  Ctrl+O   打开图像\n"
            "  Ctrl+S   保存结果\n"
            "  Ctrl+Q   退出程序",
        )

    def _about(self) -> None:
        messagebox.showinfo(
            "关于",
            "激光角点提取系统  v2.0\n\n"
            "基于 OpenCV 的激光图像角点提取软件\n"
            "支持 Harris（热力图+亚像素）、Shi-Tomasi（亚像素）、\n"
            "FAST、ORB、激光亮斑等多种算法\n\n"
            "技术栈:\n"
            "  Python 3 · OpenCV · tkinter · NumPy · Matplotlib\n\n"
            "版权所有 © 2024",
        )

    # ==================================================================
    # Session management
    # ==================================================================

    def _logout(self) -> None:
        if messagebox.askyesno("退出登录", "确定要退出登录吗？"):
            self.on_logout()

    def _quit(self) -> None:
        if messagebox.askyesno("退出程序", "确定要退出程序吗？"):
            self.root.quit()


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()

    def show_login() -> None:
        for w in root.winfo_children():
            w.destroy()
        root.title("激光角点提取系统 – 登录")
        root.resizable(False, False)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"400x320+{(sw - 400) // 2}+{(sh - 320) // 2}")
        LoginWindow(root, show_main)

    def show_main(username: str) -> None:
        for w in root.winfo_children():
            w.destroy()
        root.resizable(True, True)
        LaserCornerExtractionApp(root, username, show_login)

    show_login()
    root.mainloop()


if __name__ == "__main__":
    main()
