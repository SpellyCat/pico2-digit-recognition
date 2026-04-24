import json
import threading
import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
import serial
from PIL import Image, ImageDraw

PICO_PORT = "COM8"
BAUD_RATE = 115200
CANVAS_SIZE = 280
OUTPUT_SIZE = 28
FEATURES_TO_SEND = 784


class MnistDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST -> Pico 2 纯净边缘推理")

        self.ser = None
        self._lock = threading.Lock()
        self._busy = False

        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        self.send_btn = tk.Button(btn_frame, text="识别 (发送原始图像至 Pico)", command=self.send_data)
        self.send_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="清空画板", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(self.root, text="等待连接...", font=("Arial", 14))
        self.result_label.pack(pady=10)

        try:
            self.ser = serial.Serial(PICO_PORT, BAUD_RATE, timeout=1, write_timeout=1)
            time.sleep(1.5)
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass
            self.result_label.config(text="串口已连接，请写字")
        except Exception as e:
            messagebox.showerror("串口错误", f"无法打开串口 {PICO_PORT}，请检查是否被占用。\n{e}")
            self.ser = None

    def paint(self, event):
        r = 14
        x1, y1 = event.x - r, event.y - r
        x2, y2 = event.x + r, event.y + r
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="已清空，请重新写字")

    @staticmethod
    def _shift_zero(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
        out = np.zeros_like(arr)
        h, w = arr.shape
        y1, y2 = max(0, dy), min(h, h + dy)
        x1, x2 = max(0, dx), min(w, w + dx)
        sy1, sx1 = max(0, -dy), max(0, -dx)
        out[y1:y2, x1:x2] = arr[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
        return out

    def build_features(self):
        bbox = self.image.getbbox()
        if bbox is None: return None

        left, top, right, bottom = bbox
        pad = 22
        left, top = max(0, left - pad), max(0, top - pad)
        right, bottom = min(self.image.width, right + pad), min(self.image.height, bottom + pad)

        crop = self.image.crop((left, top, right, bottom))
        w, h = crop.size
        if w <= 0 or h <= 0: return None

        scale = 20.0 / max(w, h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas = Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), 0)
        x, y = (OUTPUT_SIZE - new_w) // 2, (OUTPUT_SIZE - new_h) // 2
        canvas.paste(resized, (x, y))

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        mask = arr > 0.03
        if mask.any():
            ys, xs = np.nonzero(mask)
            cx, cy = float(xs.mean()), float(ys.mean())
            dx, dy = int(round((OUTPUT_SIZE - 1) / 2 - cx)), int(round((OUTPUT_SIZE - 1) / 2 - cy))
            arr = self._shift_zero(arr, dy, dx)

        # 直接导出 784 个像素值，为了减少串口传输延迟保留3位小数
        flat = arr.ravel().tolist()
        return [round(x, 3) for x in flat]

    def send_data(self):
        if not self.ser: return

        with self._lock:
            if self._busy: return
            self._busy = True

        self.send_btn.config(state=tk.DISABLED)

        try:
            features = self.build_features()
            if features is None:
                self._finish_request("请先写一个数字")
                return

            try: self.ser.reset_input_buffer()
            except Exception: pass

            payload = json.dumps(features) + "\n"
            self.ser.write(payload.encode("utf-8"))
            self.ser.flush()
            self.result_label.config(text="发送图像并由 Pico 计算中...")
            threading.Thread(target=self.receive_result, daemon=True).start()
        except Exception as e:
            self._finish_request(f"错误: {e}")

    def _finish_request(self, text: str):
        def apply():
            self.result_label.config(text=text)
            self.send_btn.config(state=tk.NORMAL)
            with self._lock: self._busy = False
        self.root.after(0, apply)

    def receive_result(self):
        if not self.ser: return
        deadline = time.time() + 5.0
        try:
            while time.time() < deadline:
                raw = self.ser.readline()
                if not raw: continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("{"): continue

                data = json.loads(line)
                if "digit" in data:
                    res_text = f"Pico 识别结果: {data['digit']} (置信度 {data['confidence']:.1f}%)"
                    self._finish_request(res_text)
                    return
                if "error" in data:
                    self._finish_request(f"错误: {data['error']}")
                    return

            self._finish_request("超时未收到 Pico 响应")
        except Exception as e:
            self._finish_request(f"错误: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MnistDigitApp(root)
    root.mainloop()