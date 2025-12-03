"""Minimal Tkinter UI with live preview for AshDithEr.

Provides a desktop UI to:
- Load an image
- Adjust pixel downscale factor, dithering method, palette size, and dither stage
- See a live preview (computed on a reduced-size image for speed)
- Save the full-resolution result

This UI uses only the project's existing dependencies (Pillow, NumPy) and
the standard library (tkinter). Error-diffusion dithers benefit from the
optional Numba acceleration already included in the project.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .utils.loader import load_image, save_image
from .utils.pixelate import downscale_block_average
from .utils.upscale import upscale_nearest
from .utils.resize import resize_nearest_scale
from .dithers import apply_dither


DITHER_METHODS = [
    "none",
    "floyd",
    "atkinson",
    "burkes",
    "sierra",
    "bayer2",
    "bayer4",
    "bayer8",
]

DITHER_STAGES = [
    "after-downscale",
    "after-upscale1",
    "after-upscale2",
    "after-final",
]


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="RGB")


def _fit_preview(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    scale = min(max_side / max(w, 1), max_side / max(h, 1))
    if scale >= 1.0:
        return im.copy()
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    # Use nearest so pixel edges stay crisp in preview
    return im.resize((new_w, new_h), resample=Image.NEAREST)


@dataclass
class UIState:
    image_path: Optional[Path] = None
    image_full: Optional[np.ndarray] = None  # full-res RGB uint8
    image_preview_src: Optional[np.ndarray] = None  # reduced source for speed
    preview_max_side: int = 768
    debounce_ms: int = 150
    worker: Optional[threading.Thread] = None
    cancel_flag: bool = False


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AshDithEr UI")
        self.state = UIState()

        # Controls
        # Defaults aligned to requested screenshot
        self.var_pixel = tk.IntVar(value=16)
        self.var_colors = tk.IntVar(value=32)
        self.var_scale = tk.IntVar(value=4)
        self.var_dither = tk.StringVar(value="bayer2")
        self.var_stage = tk.StringVar(value="after-upscale2")
        self.var_dither_scale = tk.DoubleVar(value=1.0)

        self._build_ui()
        self._pending_update: Optional[str] = None
        self._preview_imgtk: Optional[ImageTk.PhotoImage] = None

    def _build_ui(self) -> None:
        frm = ttk.Frame(self.root, padding=8)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Top controls
        top = ttk.Frame(frm)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(10, weight=1)

        ttk.Button(top, text="Open…", command=self.on_open).grid(row=0, column=0, padx=(0, 8))

        ttk.Label(top, text="Pixel").grid(row=0, column=1)
        s_pixel = ttk.Spinbox(top, from_=1, to=256, textvariable=self.var_pixel, width=5, command=self.on_params_changed)
        s_pixel.grid(row=0, column=2, padx=(4, 12))

        ttk.Label(top, text="Colors").grid(row=0, column=3)
        s_colors = ttk.Spinbox(top, from_=2, to=256, textvariable=self.var_colors, width=5, command=self.on_params_changed)
        s_colors.grid(row=0, column=4, padx=(4, 12))

        ttk.Label(top, text="Dither").grid(row=0, column=5)
        cb_dither = ttk.Combobox(top, values=DITHER_METHODS, textvariable=self.var_dither, width=10, state="readonly")
        cb_dither.grid(row=0, column=6, padx=(4, 12))
        cb_dither.bind("<<ComboboxSelected>>", lambda e: self.on_params_changed())

        ttk.Label(top, text="Stage").grid(row=0, column=7)
        cb_stage = ttk.Combobox(top, values=DITHER_STAGES, textvariable=self.var_stage, width=16, state="readonly")
        cb_stage.grid(row=0, column=8, padx=(4, 12))
        cb_stage.bind("<<ComboboxSelected>>", lambda e: self.on_params_changed())

        ttk.Label(top, text="Dither scale").grid(row=0, column=9)
        s_dscale = ttk.Spinbox(top, from_=0.25, to=4.0, increment=0.25, textvariable=self.var_dither_scale, width=6, command=self.on_params_changed)
        s_dscale.grid(row=0, column=10, padx=(4, 12))

        ttk.Label(top, text="Final x").grid(row=0, column=11)
        s_scale = ttk.Spinbox(top, from_=1, to=16, textvariable=self.var_scale, width=5, command=self.on_params_changed)
        s_scale.grid(row=0, column=12, padx=(4, 12))

        ttk.Button(top, text="Save Full…", command=self.on_save_full).grid(row=0, column=13)

        # Preview area
        self.canvas = tk.Canvas(frm, bg="#222", width=800, height=600)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

    def on_open(self) -> None:
        path = filedialog.askopenfilename(title="Open image")
        if not path:
            return
        try:
            arr = load_image(path)
        except Exception as e:
            messagebox.showerror("Open failed", str(e))
            return
        self.state.image_path = Path(path)
        self.state.image_full = arr
        # Build preview source by fitting to preview size (maintain aspect)
        pil = _to_pil(arr)
        pil_small = _fit_preview(pil, self.state.preview_max_side)
        self.state.image_preview_src = np.array(pil_small, dtype=np.uint8)
        self._trigger_update()

    def on_save_full(self) -> None:
        if self.state.image_full is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", ".png"), ("All", "*.*")])
        if not out:
            return
        try:
            arr = self._run_pipeline(self.state.image_full)
            save_image(arr, out)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        messagebox.showinfo("Saved", f"Wrote {out}")

    def on_params_changed(self) -> None:
        self._trigger_update()

    def _trigger_update(self) -> None:
        # Debounce UI changes to avoid recomputing too frequently
        if self._pending_update is not None:
            self.root.after_cancel(self._pending_update)
        self._pending_update = self.root.after(self.state.debounce_ms, self._start_worker)  # type: ignore

    def _start_worker(self) -> None:
        self._pending_update = None
        if self.state.worker and self.state.worker.is_alive():
            # Signal cancel and wait a little
            self.state.cancel_flag = True
            self.state.worker.join(timeout=0.1)
        self.state.cancel_flag = False
        self.state.worker = threading.Thread(target=self._compute_preview, daemon=True)
        self.state.worker.start()

    def _compute_preview(self) -> None:
        src = self.state.image_preview_src
        if src is None:
            return
        try:
            out = self._run_pipeline(src)
        except Exception as e:
            # Render error as message on canvas. Bind the message into lambda.
            msg = str(e)
            def _cb(m=msg):
                self._show_error(m)
            self.root.after(0, _cb)
            return
        if self.state.cancel_flag:
            return
        pil = _to_pil(out)
        imgtk = ImageTk.PhotoImage(pil)
        self.root.after(0, lambda: self._update_canvas(imgtk))

    def _update_canvas(self, imgtk: ImageTk.PhotoImage) -> None:
        self._preview_imgtk = imgtk  # keep reference to prevent GC
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        iw = imgtk.width()
        ih = imgtk.height()
        x = max(0, (w - iw) // 2)
        y = max(0, (h - ih) // 2)
        self.canvas.create_image(x, y, anchor="nw", image=imgtk)

    def _show_error(self, msg: str) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(10, 10, anchor="nw", fill="#fff", text=f"Error: {msg}")

    def _run_pipeline(self, arr: np.ndarray) -> np.ndarray:
        pixel = max(1, int(self.var_pixel.get()))
        scale_final = max(1, int(self.var_scale.get()))
        method = self.var_dither.get()
        stage = self.var_stage.get()
        colors = max(2, int(self.var_colors.get()))
        d_scale = float(self.var_dither_scale.get())

        # Downscale
        img_ds = downscale_block_average(arr, pixel) if pixel > 1 else arr.copy()
        # Upscale x2
        img_up1 = upscale_nearest(img_ds, 2)
        # Prepare second x2 and optional final scale
        img_up2 = upscale_nearest(img_up1, 2)
        img_final_pre = upscale_nearest(img_up2, scale_final) if scale_final > 1 else img_up2

        if stage == "after-downscale":
            base = img_ds
            if d_scale != 1.0:
                tmp = resize_nearest_scale(base, d_scale)
                tmp = apply_dither(tmp, num_colors=colors, method=method)
                work = resize_nearest_scale(tmp, 1.0 / d_scale)
            else:
                work = apply_dither(base, num_colors=colors, method=method)
            work = upscale_nearest(work, 2)
            work = upscale_nearest(work, 2)
            if scale_final > 1:
                work = upscale_nearest(work, scale_final)
        elif stage == "after-upscale1":
            base = img_up1
            if d_scale != 1.0:
                tmp = resize_nearest_scale(base, d_scale)
                tmp = apply_dither(tmp, num_colors=colors, method=method)
                work = resize_nearest_scale(tmp, 1.0 / d_scale)
            else:
                work = apply_dither(base, num_colors=colors, method=method)
            work = upscale_nearest(work, 2)
            if scale_final > 1:
                work = upscale_nearest(work, scale_final)
        elif stage == "after-upscale2":
            base = img_up2
            if d_scale != 1.0:
                tmp = resize_nearest_scale(base, d_scale)
                tmp = apply_dither(tmp, num_colors=colors, method=method)
                work = resize_nearest_scale(tmp, 1.0 / d_scale)
            else:
                work = apply_dither(base, num_colors=colors, method=method)
            if scale_final > 1:
                work = upscale_nearest(work, scale_final)
        else:  # after-final
            base = img_final_pre
            if d_scale != 1.0:
                tmp = resize_nearest_scale(base, d_scale)
                tmp = apply_dither(tmp, num_colors=colors, method=method)
                work = resize_nearest_scale(tmp, 1.0 / d_scale)
            else:
                work = apply_dither(base, num_colors=colors, method=method)
        return work


def run_ui() -> None:
    root = tk.Tk()
    # Use system default styling where available
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:
        pass
    App(root)
    root.minsize(640, 480)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    run_ui()
