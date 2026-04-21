import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFilter
import serial
import json
import threading
import time
import numpy as np

PICO_PORT = 'COM8'
BAUD_RATE = 115200


class DistributedDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pico 2 边缘计算节点")

        self.ser = None
        self._lock = threading.Lock()
        self._busy = False

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='black', cursor="cross")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.image = Image.new("L", (200, 200), 0)
        self.draw = ImageDraw.Draw(self.image)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        self.send_btn = tk.Button(btn_frame, text="识别 (发送至 Pico)", command=self.send_data)
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
        r = 10
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="已清空，请重新写字")

    def _center_of_mass(self, arr):
        total = float(arr.sum())
        if total <= 1e-8:
            return None
        ys, xs = np.indices(arr.shape)
        cx = float((xs * arr).sum() / total)
        cy = float((ys * arr).sum() / total)
        return cx, cy

    def _deskew(self, img):
        arr = np.asarray(img, dtype=np.float32)
        com = self._center_of_mass(arr)
        if com is None:
            return img

        cx, cy = com
        ys, xs = np.indices(arr.shape)
        total = float(arr.sum())
        mu02 = float((((ys - cy) ** 2) * arr).sum() / total)
        mu11 = float((((xs - cx) * (ys - cy)) * arr).sum() / total)
        if mu02 <= 1e-6:
            return img

        skew = mu11 / mu02
        matrix = (1.0, -skew, skew * cy, 0.0, 1.0, 0.0)
        return img.transform(img.size, Image.Transform.AFFINE, matrix, resample=Image.Resampling.BICUBIC)

    def _shift_center(self, img):
        arr = np.asarray(img, dtype=np.float32)
        com = self._center_of_mass(arr)
        if com is None:
            return img

        cx, cy = com
        dx = (img.width - 1) / 2.0 - cx
        dy = (img.height - 1) / 2.0 - cy
        matrix = (1.0, 0.0, -dx, 0.0, 1.0, -dy)
        return img.transform(img.size, Image.Transform.AFFINE, matrix, resample=Image.Resampling.BICUBIC)

    def build_features(self):
        bbox = self.image.getbbox()
        if bbox is None:
            return None

        left, top, right, bottom = bbox
        pad = 22
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(self.image.width, right + pad)
        bottom = min(self.image.height, bottom + pad)

        crop = self.image.crop((left, top, right, bottom))
        w, h = crop.size
        side = max(w, h)

        square = Image.new("L", (side, side), 0)
        offset = ((side - w) // 2, (side - h) // 2)
        square.paste(crop, offset)

        square = self._deskew(square)
        square = self._shift_center(square)
        square = square.filter(ImageFilter.GaussianBlur(radius=0.35))
        img_resized = square.resize((8, 8), Image.Resampling.LANCZOS)

        return [round((p / 255.0) * 16.0, 4) for p in img_resized.getdata()]

    def send_data(self):
        if not self.ser:
            return

        with self._lock:
            if self._busy:
                return
            self._busy = True

        self.send_btn.config(state=tk.DISABLED)

        try:
            features = self.build_features()
            if features is None:
                self._finish_request("请先写一个数字")
                return

            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass

            payload = json.dumps(features) + "\n"
            self.ser.write(payload.encode("utf-8"))
            self.ser.flush()
            self.result_label.config(text="计算中...")
            threading.Thread(target=self.receive_result, daemon=True).start()

        except Exception as e:
            self._finish_request(f"错误: {e}")

    def _finish_request(self, text: str):
        def apply():
            self.result_label.config(text=text)
            self.send_btn.config(state=tk.NORMAL)
            with self._lock:
                self._busy = False
        self.root.after(0, apply)

    def receive_result(self):
        if not self.ser:
            self._finish_request("串口未连接")
            return

        deadline = time.time() + 4.0
        try:
            while time.time() < deadline:
                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("{"):
                    continue

                data = json.loads(line)
                if "digit" in data:
                    res_text = f"识别结果: {data['digit']} (置信度 {data['confidence']:.1f}%)"
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
    app = DistributedDigitApp(root)
    root.mainloop()
