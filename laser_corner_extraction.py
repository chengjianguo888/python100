"""
激光角点提取系统
Laser Corner Extraction System

一个基于 OpenCV 的激光图像角点提取 GUI 软件，支持多种角点检测与边缘检测算法。

使用方法:
    pip install -r requirements.txt
    python laser_corner_extraction.py

默认账户:
    用户名: admin  密码: admin123
    用户名: user   密码: user123
"""

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
# User credentials: values are SHA-256 hashes of the passwords.
# (In production use a proper auth backend with bcrypt/argon2 + salts.)
# ---------------------------------------------------------------------------
USERS = {
    "admin": _hash_password("admin123"),
    "user":  _hash_password("user123"),
}


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
        self._tk_img_orig = None   # PhotoImage reference (prevents GC)
        self._tk_img_proc = None

        self.root.title(f"激光角点提取系统  –  当前用户: {username}")
        self.root.geometry("1280x820")

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
        pm.add_separator()
        pm.add_command(label="Canny 边缘检测", command=self._canny)
        pm.add_command(label="清除结果", command=self._clear)

        # View
        vm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="视图", menu=vm)
        vm.add_command(label="图像直方图", command=self._histogram)
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
        left = tk.Frame(outer, width=290, bg="#ecf0f1", bd=1, relief=tk.SUNKEN)
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

        # — Algorithm selector —
        af = tk.LabelFrame(left, text="角点检测", font=("Arial", 10), bg="#ecf0f1", padx=5, pady=5)
        af.pack(fill=tk.X, padx=6, pady=6)

        tk.Label(af, text="检测算法:", bg="#ecf0f1", font=("Arial", 9)).pack(anchor="w")
        self._algo_var = tk.StringVar(value="Harris")
        algo_cb = ttk.Combobox(
            af,
            textvariable=self._algo_var,
            values=["Harris", "Shi-Tomasi", "FAST"],
            state="readonly",
            width=24,
        )
        algo_cb.pack(pady=(3, 5))
        algo_cb.bind("<<ComboboxSelected>>", self._on_algo_change)

        # Harris params
        self._pf_harris = tk.LabelFrame(af, text="Harris 参数", bg="#ecf0f1", font=("Arial", 9), padx=4, pady=4)
        _row = 0
        for label, var_name, default, lo, hi, dtype in [
            ("块大小:", "_p_block",  2,    1, 15,  int),
            ("K 值:",   "_p_k",     0.04, 0.0,  1.0, float),
            ("阈  值:", "_p_thresh", 0.01, 0.0,  1.0, float),
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
        tk.Button(rf, text="保存结果", command=self._save_result,
                  font=("Arial", 10), bg="#9b59b6", fg="white", width=22).pack(pady=3)
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

    def _on_algo_change(self, _event) -> None:
        for frame in (self._pf_harris, self._pf_shi, self._pf_fast):
            frame.pack_forget()
        algo = self._algo_var.get()
        if algo == "Harris":
            self._pf_harris.pack(fill=tk.X, pady=4)
        elif algo == "Shi-Tomasi":
            self._pf_shi.pack(fill=tk.X, pady=4)
        elif algo == "FAST":
            self._pf_fast.pack(fill=tk.X, pady=4)

    def _on_mouse(self, event) -> None:
        if self._src_img is not None:
            self._coord_var.set(f"坐标: ({event.x}, {event.y})")

    def _show_image(self, img: np.ndarray, canvas: tk.Canvas, store_attr: str) -> None:
        """Fit *img* (BGR or GRAY) into *canvas* and keep a PhotoImage reference."""
        canvas.delete("all")
        canvas.update_idletasks()
        cw = canvas.winfo_width() or 900
        ch = canvas.winfo_height() or 650

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

    # ==================================================================
    # Detection algorithms
    # ==================================================================

    def _run_detection(self) -> None:
        self._detect(self._algo_var.get())

    def _detect(self, algorithm: str) -> None:
        if self._src_img is None:
            messagebox.showwarning("提示", "请先打开一张图像。")
            return
        try:
            gray = cv2.cvtColor(self._src_img, cv2.COLOR_BGR2GRAY)
            result = self._src_img.copy()
            corners = []

            if algorithm == "Harris":
                block_size = int(self._p_block.get())
                k = float(self._p_k.get())
                thresh = float(self._p_thresh.get())
                gray_f = np.float32(gray)
                dst = cv2.cornerHarris(gray_f, block_size, 3, k)
                dst = cv2.dilate(dst, None)
                mask = dst > thresh * dst.max()
                result[mask] = [0, 0, 255]
                ys, xs = np.where(mask)
                corners = list(zip(xs.tolist(), ys.tolist()))

            elif algorithm == "Shi-Tomasi":
                max_c = int(self._p_max_corners.get())
                quality = float(self._p_quality.get())
                min_d = int(self._p_min_dist.get())
                pts = cv2.goodFeaturesToTrack(gray, max_c, quality, min_d)
                if pts is not None:
                    for pt in np.intp(pts):
                        x, y = pt.ravel()
                        cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
                        corners.append((int(x), int(y)))

            elif algorithm == "FAST":
                fast = cv2.FastFeatureDetector_create(
                    threshold=int(self._p_fast_thresh.get()),
                    nonmaxSuppression=bool(self._p_nonmax.get()),
                )
                kps = fast.detect(gray, None)
                for kp in kps:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    cv2.circle(result, (x, y), 4, (255, 0, 0), -1)
                    corners.append((x, y))

            self._result_img = result
            self._corners = corners
            self._show_image(result, self._canvas_proc, "_tk_img_proc")
            self._count_var.set(f"角点数量: {len(corners)}")
            self._status(f"{algorithm} 角点检测完成 – 检测到 {len(corners)} 个角点")
            self._nb.select(1)

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
            self._show_image(edges, self._canvas_proc, "_tk_img_proc")
            count = int(np.count_nonzero(edges))
            self._status(f"Canny 边缘检测完成 – 边缘像素数: {count}")
            self._nb.select(1)
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
            "激光角点提取系统 – 使用说明\n\n"
            "1. 打开图像：点击「打开图像」或使用 Ctrl+O\n"
            "2. 选择算法：从下拉菜单选择角点检测算法\n"
            "3. 调整参数：根据需要修改算法参数\n"
            "4. 执行检测：点击「执行角点检测」按钮\n"
            "5. 查看结果：在「处理结果」标签页查看\n"
            "6. 保存结果：点击「保存结果」或使用 Ctrl+S\n\n"
            "支持的算法：\n"
            "• Harris       – 经典角点检测算法\n"
            "• Shi-Tomasi   – 改进的 Harris 算法\n"
            "• FAST         – 高速角点检测算法\n"
            "• Canny        – 边缘提取算法\n\n"
            "快捷键：\n"
            "  Ctrl+O   打开图像\n"
            "  Ctrl+S   保存结果\n"
            "  Ctrl+Q   退出程序",
        )

    def _about(self) -> None:
        messagebox.showinfo(
            "关于",
            "激光角点提取系统  v1.0\n\n"
            "基于 OpenCV 的激光图像角点提取软件\n"
            "支持 Harris、Shi-Tomasi、FAST 等多种算法\n\n"
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
