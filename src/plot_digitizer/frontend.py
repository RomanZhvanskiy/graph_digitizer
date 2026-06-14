"""Tk frontend for Plot Digitizer.

Frontend responsibilities:
- open image
- remove existing strict-red pixels from the working image
- allow user to mark any three of four plot corners
- collect axis bounds as text inputs
- save a temporary marked image
- call backend digitization
"""
from __future__ import annotations

import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageTk

from .core import DigitizeOptions, DigitizeResult, digitize_marked_image

Point2D = Tuple[float, float]

STRICT_RED = (255, 0, 0)
BLACK = (0, 0, 0)
MARKER_RADIUS_PX = 8


class PlotDigitizerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Plot Digitizer")
        self.geometry("1280x820")
        self.minsize(1000, 650)

        self.image_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.cleaned_image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0

        self.pending_corner: str | None = None
        self.result: DigitizeResult | None = None

        self.status_text = tk.StringVar(value="Open an image to start.")

        self.corner_enabled: Dict[str, tk.BooleanVar] = {
            "tl": tk.BooleanVar(value=True),
            "tr": tk.BooleanVar(value=False),
            "bl": tk.BooleanVar(value=True),
            "br": tk.BooleanVar(value=True),
        }

        self.corner_points: Dict[str, Point2D | None] = {
            "tl": None,
            "tr": None,
            "bl": None,
            "br": None,
        }

        self.corner_coord_vars: Dict[str, tk.StringVar] = {}
        self.corner_checkbuttons: Dict[str, ttk.Checkbutton] = {}

        self.bound_vars = {
            "xmin": tk.StringVar(value="0"),
            "xmax": tk.StringVar(value="1"),
            "ymin": tk.StringVar(value="0"),
            "ymax": tk.StringVar(value="1"),
        }

        self.option_vars = {
            "auto_colors": tk.BooleanVar(value=False),
            "exclude_grayscale": tk.BooleanVar(value=True),
            "xscale_log": tk.BooleanVar(value=False),
            "yscale_log": tk.BooleanVar(value=False),
            "monotone_y": tk.BooleanVar(value=False),
            "keep_outside_axes": tk.BooleanVar(value=False),
            "debug": tk.BooleanVar(value=False),
            "debug_images": tk.BooleanVar(value=False),
            "k": tk.StringVar(value="100"),
            "num_colors": tk.StringVar(value="0"),
            "min_sat": tk.StringVar(value="30"),
            "min_val": tk.StringVar(value="30"),
            "knn_k": tk.StringVar(value="3"),
            "knn_thresh": tk.StringVar(value="35.0"),
        }

        self._build_ui()
        self.on_corner_tick_changed()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(root)
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="Open image", command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Digitize", command=self.digitize_clicked).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(toolbar, textvariable=self.status_text).pack(side=tk.LEFT, padx=(16, 0))

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        left = ttk.Frame(body)
        right = ttk.Frame(body, width=460)
        body.add(left, weight=4)
        body.add(right, weight=1)

        self.canvas = tk.Canvas(left, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda _event: self.redraw_image())

        self._build_controls(right)

    def _build_controls(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        corner_tab = ttk.Frame(notebook, padding=8)
        opt_tab = ttk.Frame(notebook, padding=8)
        res_tab = ttk.Frame(notebook, padding=8)

        notebook.add(corner_tab, text="Corners")
        notebook.add(opt_tab, text="Options")
        notebook.add(res_tab, text="Result")

        ttk.Label(
            corner_tab,
            text="Tick exactly three plot corners, then press Select point and click the image once.",
            wraplength=400,
        ).pack(anchor=tk.W, pady=(0, 8))

        for key, label in [
            ("tl", "Top-left"),
            ("tr", "Top-right"),
            ("bl", "Bottom-left"),
            ("br", "Bottom-right"),
        ]:
            self._corner_row(corner_tab, key, label)

        ttk.Separator(corner_tab).pack(fill=tk.X, pady=10)

        bounds = ttk.LabelFrame(corner_tab, text="Axis bounds", padding=8)
        bounds.pack(fill=tk.X)

        self._bound_row(bounds, "xmin", "x min", 0)
        self._bound_row(bounds, "xmax", "x max", 1)
        self._bound_row(bounds, "ymin", "y min", 2)
        self._bound_row(bounds, "ymax", "y max", 3)

        ttk.Label(
            corner_tab,
            text=(
                "Backend mapping: bottom-left=(xmin,ymin), bottom-right=(xmax,ymin), "
                "top-left=(xmin,ymax). The missing fourth corner is inferred."
            ),
            wraplength=400,
        ).pack(anchor=tk.W, pady=(10, 0))

        self._option_checkbox(opt_tab, "Auto-detect multiple colour curves", "auto_colors")
        self._option_checkbox(opt_tab, "Exclude grayscale pixels", "exclude_grayscale")
        self._option_checkbox(opt_tab, "Log x-axis", "xscale_log")
        self._option_checkbox(opt_tab, "Log y-axis", "yscale_log")
        self._option_checkbox(opt_tab, "Force monotone non-increasing y", "monotone_y")
        self._option_checkbox(opt_tab, "Keep points outside axes", "keep_outside_axes")
        self._option_checkbox(opt_tab, "Debug", "debug")
        self._option_checkbox(opt_tab, "Debug images", "debug_images")

        ttk.Separator(opt_tab).pack(fill=tk.X, pady=8)

        for label, key in [
            ("Representative points per curve", "k"),
            ("Force number of colours/clusters, 0 = auto", "num_colors"),
            ("Minimum saturation", "min_sat"),
            ("Minimum brightness/value", "min_val"),
            ("k-NN neighbours", "knn_k"),
            ("k-NN threshold", "knn_thresh"),
        ]:
            row = ttk.Frame(opt_tab)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=self.option_vars[key], width=10).pack(side=tk.RIGHT)

        self.result_text = tk.Text(res_tab, height=12, wrap=tk.NONE)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _corner_row(self, parent: ttk.Frame, key: str, label: str) -> None:
        frame = ttk.LabelFrame(parent, text=label, padding=6)
        frame.pack(fill=tk.X, pady=(0, 8))
    
        top = ttk.Frame(frame)
        top.pack(fill=tk.X)
    
        checkbox = ttk.Checkbutton(
            top,
            text="Use this corner",
            variable=self.corner_enabled[key],
            command=self.on_corner_tick_changed,
        )
        checkbox.pack(side=tk.LEFT)
        self.corner_checkbuttons[key] = checkbox
    
        ttk.Button(
            top,
            text="Select point",
            command=lambda k=key: self.start_select_corner(k),
        ).pack(side=tk.RIGHT)
    
        coord_var = tk.StringVar(value="No point selected.")
        self.corner_coord_vars[key] = coord_var
        ttk.Label(frame, textvariable=coord_var).pack(anchor=tk.W, pady=(4, 0))


    def _bound_row(self, parent: ttk.Frame, key: str, label: str, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=2)
        ttk.Entry(parent, textvariable=self.bound_vars[key], width=14).grid(row=row, column=1, sticky=tk.EW, pady=2)
        parent.columnconfigure(1, weight=1)

    def _option_checkbox(self, parent: ttk.Frame, text: str, key: str) -> None:
        ttk.Checkbutton(parent, text=text, variable=self.option_vars[key]).pack(anchor=tk.W)

    def enabled_corners(self) -> list[str]:
        return [key for key, var in self.corner_enabled.items() if bool(var.get())]

    def on_corner_tick_changed(self) -> None:
        enabled = self.enabled_corners()
    
        if len(enabled) > 3:
            # Keep only the first three enabled corners.
            for key in ["tl", "tr", "bl", "br"]:
                if key in enabled[3:]:
                    self.corner_enabled[key].set(False)
                    self.corner_points[key] = None
    
        enabled = self.enabled_corners()
    
        for key, checkbox in self.corner_checkbuttons.items():
            if len(enabled) >= 3 and key not in enabled:
                checkbox.state(["disabled"])
            else:
                checkbox.state(["!disabled"])
    
            if key not in enabled:
                self.corner_points[key] = None
    
        self._update_corner_labels()
        self.redraw_image()
        

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Open plot image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not filename:
            return

        self.image_path = Path(filename)
        self.original_image = Image.open(self.image_path).convert("RGB")
        self.cleaned_image = remove_strict_red_pixels(self.original_image)

        self.corner_points = {key: None for key in self.corner_points}
        self.result = None
        self.result_text.delete("1.0", tk.END)
        self._update_corner_labels()

        self.status_text.set(f"Loaded {self.image_path.name}; original strict-red pixels converted to black.")
        self.redraw_image()

    def start_select_corner(self, key: str) -> None:
        if self.cleaned_image is None:
            messagebox.showerror("No image", "Open an image first.")
            return

        if not self.corner_enabled[key].get():
            messagebox.showerror("Corner not enabled", "Tick this corner before selecting it.")
            return

        if len(self.enabled_corners()) != 3:
            messagebox.showerror("Corner selection", "Tick exactly three corners before selecting points.")
            return

        self.pending_corner = key
        self.status_text.set(f"Click once on the image to place {key.upper()} marker.")

    def on_canvas_click(self, event: tk.Event) -> None:
        if self.cleaned_image is None or self.display_scale <= 0:
            return

        if self.pending_corner is None:
            self.status_text.set("Press a corner's Select point button before clicking.")
            return

        x = event.x / self.display_scale
        y = event.y / self.display_scale

        x = min(max(x, 0.0), float(self.cleaned_image.width - 1))
        y = min(max(y, 0.0), float(self.cleaned_image.height - 1))

        corner = self.pending_corner
        self.corner_points[corner] = (x, y)
        self.pending_corner = None

        self._update_corner_labels()
        self.status_text.set(f"Set {corner.upper()} marker at ({x:.1f}, {y:.1f}).")
        self.redraw_image()

    def _update_corner_labels(self) -> None:
        for key, var in self.corner_coord_vars.items():
            pt = self.corner_points.get(key)
            if pt is None:
                var.set("No point selected.")
            else:
                var.set(f"Pixel: x={pt[0]:.3f}, y={pt[1]:.3f}")

    def marked_image(self) -> Image.Image:
        if self.cleaned_image is None:
            raise ValueError("No image loaded.")

        image = self.cleaned_image.copy()
        draw = ImageDraw.Draw(image)

        for key in self.enabled_corners():
            point = self.corner_points.get(key)
            if point is None:
                continue
            x, y = point
            r = MARKER_RADIUS_PX
            draw.ellipse((x - r, y - r, x + r, y + r), fill=STRICT_RED)

        return image

    def redraw_image(self) -> None:
        if self.cleaned_image is None:
            return

        marked = self.marked_image()

        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)

        scale = min(canvas_w / marked.width, canvas_h / marked.height, 1.0)
        self.display_scale = scale

        new_w = max(1, int(marked.width * scale))
        new_h = max(1, int(marked.height * scale))

        self.display_image = marked.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        self._draw_corner_labels()

    def _draw_corner_labels(self) -> None:
        for key, label in [("tl", "TL"), ("tr", "TR"), ("bl", "BL"), ("br", "BR")]:
            point = self.corner_points.get(key)
            if point is None:
                continue

            x = point[0] * self.display_scale
            y = point[1] * self.display_scale

            self.canvas.create_text(
                x + 12,
                y - 12,
                text=label,
                fill="#ff4444",
                anchor=tk.W,
                font=("TkDefaultFont", 11, "bold"),
            )

    def parse_bounds(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, var in self.bound_vars.items():
            try:
                out[key] = float(var.get())
            except ValueError as exc:
                raise ValueError(f"{key} must be a number.") from exc

        if out["xmin"] == out["xmax"]:
            raise ValueError("xmin and xmax must differ.")
        if out["ymin"] == out["ymax"]:
            raise ValueError("ymin and ymax must differ.")

        return out

    def parse_options(self) -> DigitizeOptions:
        def int_value(key: str) -> int:
            try:
                return int(self.option_vars[key].get())
            except ValueError as exc:
                raise ValueError(f"Option {key} must be an integer.") from exc

        def float_value(key: str) -> float:
            try:
                return float(self.option_vars[key].get())
            except ValueError as exc:
                raise ValueError(f"Option {key} must be a number.") from exc

        return DigitizeOptions(
            k=int_value("k"),
            min_sat=int_value("min_sat"),
            min_val=int_value("min_val"),
            knn_k=int_value("knn_k"),
            knn_thresh=float_value("knn_thresh"),
            auto_colors=bool(self.option_vars["auto_colors"].get()),
            exclude_grayscale=bool(self.option_vars["exclude_grayscale"].get()),
            num_colors=int_value("num_colors"),
            xscale_log=bool(self.option_vars["xscale_log"].get()),
            yscale_log=bool(self.option_vars["yscale_log"].get()),
            monotone_y=bool(self.option_vars["monotone_y"].get()),
            keep_outside_axes=bool(self.option_vars["keep_outside_axes"].get()),
            debug=bool(self.option_vars["debug"].get()),
            debug_images=bool(self.option_vars["debug_images"].get()),
        )

    def validate_markers(self) -> None:
        enabled = self.enabled_corners()
        if len(enabled) != 3:
            raise ValueError("Tick exactly three corners.")

        missing = [key.upper() for key in enabled if self.corner_points.get(key) is None]
        if missing:
            raise ValueError(f"Select marker point(s): {', '.join(missing)}.")

    def digitize_clicked(self) -> None:
        if self.cleaned_image is None:
            messagebox.showerror("No image", "Open an image first.")
            return

        try:
            self.validate_markers()
            bounds = self.parse_bounds()
            options = self.parse_options()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.status_text.set("Digitizing...")
        self.result_text.delete("1.0", tk.END)

        marked = self.marked_image()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        marked.save(tmp_path)

        thread = threading.Thread(
            target=self._digitize_worker,
            args=(tmp_path, bounds, options),
            daemon=True,
        )
        thread.start()

    def _digitize_worker(self, image_path: Path, bounds: dict[str, float], options: DigitizeOptions) -> None:
        try:
            result = digitize_marked_image(
                image_path,
                xmin=bounds["xmin"],
                xmax=bounds["xmax"],
                ymin=bounds["ymin"],
                ymax=bounds["ymax"],
                options=options,
            )
        except Exception as exc:
            self.after(0, lambda: self._digitize_failed(exc))
            return

        self.after(0, lambda: self._digitize_finished(result))

    def _digitize_failed(self, exc: Exception) -> None:
        self.status_text.set("Digitization failed.")
        messagebox.showerror("Digitization failed", str(exc))

    def _digitize_finished(self, result: DigitizeResult) -> None:
        self.result = result
        df = result.to_dataframe()

        self.result_text.delete("1.0", tk.END)

        for warning in result.warnings:
            self.result_text.insert(tk.END, f"WARNING: {warning}\n")
        if result.warnings:
            self.result_text.insert(tk.END, "\n")

        self.result_text.insert(tk.END, df.head(40).to_string(index=False))
        if len(df) > 40:
            self.result_text.insert(tk.END, f"\n\n... showing first 40 of {len(df)} rows")

        self.status_text.set(f"Digitized {len(result.curves)} curve(s), {len(df)} row(s).")

    def save_csv(self) -> None:
        if self.result is None:
            messagebox.showerror("No result", "Digitize an image first.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="digitized_curves.csv",
        )
        if not filename:
            return

        self.result.to_csv(filename)
        self.status_text.set(f"Saved {filename}")


def remove_strict_red_pixels(image: Image.Image) -> Image.Image:
    """Convert original strict-red pixels to black.

    Must match backend strict-red threshold:
        R >= 240, G <= 10, B <= 10
    """
    rgb = image.convert("RGB")
    pixels = rgb.load()

    width, height = rgb.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r >= 240 and g <= 10 and b <= 10:
                pixels[x, y] = BLACK

    return rgb


def run() -> None:
    app = PlotDigitizerApp()
    app.mainloop()
