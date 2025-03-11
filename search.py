import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import pickle
import json
import io


class ToolTip:

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text, bg="#ffffe0", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class ImageSearchApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Image Search")
        self.root.geometry("1200x800")  # Adjust window size

        # Styling
        self.root.configure(bg="#f0f0f0")

        # Device selection
        self.device = None
        self.select_device_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.select_device_frame.pack(pady=10)

        self.device_label = tk.Label(self.select_device_frame, text="Select Device (CPU or CUDA):", font=("Helvetica", 12), bg="#f0f0f0")
        self.device_label.pack(side=tk.LEFT, padx=5)

        self.device_combobox = ttk.Combobox(self.select_device_frame, width=20, font=("Helvetica", 12))
        # 检查torch包是否安装完成
        try:
            import torch
            self.device_combobox['values'] = ['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        except Exception as e:
            self.device_combobox['values'] = ['cpu']
        self.device_combobox.set('cpu')  # Default to CPU
        self.device_combobox.pack(side=tk.LEFT, padx=5)

        self.device_button = tk.Button(self.select_device_frame, text="Set Device", font=("Helvetica", 12), bg="#4caf50", fg="white", command=self.set_device)
        self.device_button.pack(side=tk.LEFT, padx=5)
        self.device = 'cpu'  # Default to CPU

        # Search frame
        self.search_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.search_frame.pack(pady=10)

        self.label = tk.Label(self.search_frame, text="Enter Keyword:", font=("Helvetica", 12), bg="#f0f0f0")
        self.label.pack(side=tk.LEFT, padx=5)

        self.entry = tk.Entry(self.search_frame, width=30, font=("Helvetica", 12))
        self.entry.pack(side=tk.LEFT, padx=5)

        self.search_button = tk.Button(self.search_frame, text="Search", font=("Helvetica", 12), bg="#4caf50", fg="white", command=self.search_images)
        self.search_button.pack(side=tk.LEFT, padx=5)

        # Adding a secondary search box for bge-m3
        self.label_bge = tk.Label(self.search_frame, text="bge-m3 Search:", font=("Helvetica", 12), bg="#f0f0f0")
        self.label_bge.pack(side=tk.LEFT, padx=5)

        self.entry_bge = tk.Entry(self.search_frame, width=30, font=("Helvetica", 12))
        self.entry_bge.pack(side=tk.LEFT, padx=5)

        # 添加上下文复选框
        self.context_var = tk.BooleanVar()
        self.context_check = tk.Checkbutton(self.search_frame, text="使用上下文", font=("Helvetica", 10), bg="#f0f0f0", variable=self.context_var)
        self.context_check.pack(side=tk.LEFT, padx=2)

        self.tooltip_label = tk.Label(self.search_frame, text="?", font=("Helvetica", 10, "bold"), fg="blue", bg="#f0f0f0", cursor="question_arrow")
        self.tooltip_label.pack(side=tk.LEFT, padx=2)
        ToolTip(self.tooltip_label, "启用上下文时，搜索词会被格式化为'{关键词}的最佳回复是什么'，\n用于优化上下文相关的搜索结果。")

        self.search_bge_button = tk.Button(self.search_frame, text="Search", font=("Helvetica", 12), bg="#2196f3", fg="white", command=self.search_bge_images)
        self.search_bge_button.pack(side=tk.LEFT, padx=5)

        # Display frame
        self.display_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        self.listbox_frame = tk.Frame(self.display_frame, bg="#f0f0f0")
        self.listbox_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.listbox = tk.Listbox(self.listbox_frame, width=50, height=20, font=("Helvetica", 10), bg="white", selectbackground="#4caf50")
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.display_image)

        self.scrollbar = tk.Scrollbar(self.listbox_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=self.scrollbar.set)

        self.image_frame = tk.Frame(self.display_frame, bg="#f0f0f0")
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame, text="No image selected", font=("Helvetica", 14), bg="#e0e0e0", width=60, height=15, anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.cached_file_fp16_tensor = None
        self.image_name = None
        self.model = None  # Start with no model loaded
        with open("onlyvv-result-no-repeat.json", "r", encoding="utf-8") as f:
            self.onlyvv_result = json.load(f)
        self.reverse_onlyvv_result = {v: k for k, v in self.onlyvv_result.items()}
        self.onlyvv_result_keys = list(self.onlyvv_result.keys())
        self.onlyvv_result_values = list(self.onlyvv_result.values())
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="复制图片", command=self.copy_image)
        self.image_label.bind("<Button-3>", self.show_context_menu)

    def show_context_menu(self, event):
        # 检查是否有图片显示
        if self.current_image:
            try:
                self.context_menu.post(event.x_root, event.y_root)
            finally:
                self.context_menu.grab_release()

    def copy_image(self):
        if self.current_image:
            # 将PIL Image转换为Tkinter PhotoImage
            img = self.current_image

            # 将图片保存到内存缓冲区
            # 将图片转换为BMP格式
            output = io.BytesIO()
            img.convert("RGB").save(output, "BMP")
            data = output.getvalue()[14:]  # 去除BMP头
            output.close()
            import win32clipboard
            # 复制到剪贴板
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()

    def set_device(self):
        selected_device = self.device_combobox.get()
        if selected_device:
            self.device = selected_device
            messagebox.showinfo("Info", f"Device set to {selected_device}")
        else:
            messagebox.showwarning("Warning", "Please select a valid device.")

    def set_model(self):
        selected_model = self.model_combobox.get()
        if selected_model:
            self.model_id = selected_model

    def search_images(self):
        keyword = self.entry.get().strip()
        if not keyword:
            messagebox.showwarning("Warning", "Please enter a keyword.")
            return

        self.listbox.delete(0, tk.END)
        matched_ocrs = []

        for file, ocr in self.onlyvv_result.items():
            if keyword in ocr.lower() and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = file
                matched_ocrs.append(ocr)

        # Show top 20 results
        for ocr in matched_ocrs:
            self.listbox.insert(tk.END, ocr)

        if self.listbox.size() == 0:
            messagebox.showinfo("Info", "No images found with the given keyword.")

    def search_bge_images(self):

        keyword = self.entry_bge.get().strip()
        if not keyword:
            messagebox.showwarning("Warning", "Please enter a keyword for bge-m3 search.")
            return
        if self.context_var.get():
            keyword = f"{keyword}的最佳回复是什么？"

        def compute_topk_cosine_similarity(cached_file_fp16_array, embeddings, top_k=20):
            # Ensure embeddings is 2D
            embeddings = np.expand_dims(embeddings, axis=0)  # Shape: (1, D)

            # Compute cosine similarity
            cosine_sim_between_cached_files = np.dot(cached_file_fp16_array, embeddings.T).squeeze(1)

            # Get top-k indices
            indices = np.argsort(-cosine_sim_between_cached_files)[:top_k]

            return indices

        # Load the model only if the checkbox is selected
        if self.model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel

                self.cached_file_fp16_dict_np = np.load("cached_file.npy", allow_pickle=True).item()
                self.cached_file_fp16_dict_values = np.array(list(self.cached_file_fp16_dict_np.values()))
                self.cached_file_fp16_dict_keys = list(self.cached_file_fp16_dict_np.keys())
                # self.image_name = torch.load("name.pth")
                if os.path.exists("bge-m3"):
                    self.model = BGEM3FlagModel('bge-m3', use_fp16=True, devices=self.device)
                else:
                    self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=self.device)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load AI model: {e}")
                return

        # Perform the AI search using bge-m3 model
        self.listbox.delete(0, tk.END)

        embeddings = self.model.encode(
            keyword,
            batch_size=12,
            max_length=8192,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
        )['dense_vecs']
        indices = compute_topk_cosine_similarity(self.cached_file_fp16_dict_values, embeddings, top_k=40)

        matched_ocrs = [self.onlyvv_result_values[i.item()] for i in indices]

        # Show top 20 results
        for ocr in matched_ocrs:
            self.listbox.insert(tk.END, ocr)

        if self.listbox.size() == 0:
            messagebox.showinfo("Info", "No images found with the given keyword.")

    def display_image(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return

        selected_ocr = self.listbox.get(selection[0])
        try:
            img = Image.open(self.reverse_onlyvv_result[selected_ocr])
            img = img.resize((960, 540), Image.Resampling.LANCZOS)  # Resize to 960x540 using LANCZOS
            self.current_image = img
            img_tk = ImageTk.PhotoImage(img)

            self.image_label.config(image=img_tk, text="")  # Update the image
            self.image_label.image = img_tk  # Keep a reference to avoid garbage collection
        except Exception as e:
            self.image_label.config(text="Error displaying image")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()
