import os
import sys
import json
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flac_to_wav_config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"設定保存エラー: {e}")

class FlacToWavApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FLAC → WAV 単曲変換ツール")
        self.geometry("560x340")
        self.resizable(False, False)

        self.config_data = load_config()
        self.current_folder = tk.StringVar(value=self.config_data.get("last_folder", os.getcwd()))

        self.selected_file_name = tk.StringVar(value="未選択")
        self.output_filename = tk.StringVar(value="")

        self.file_map = {}  # {表示名: フルパス}

        self.setup_ui()
        self.refresh_flac_list()

    def setup_ui(self):
        pad_opts = {'padx': 12, 'pady': 6}

        # 1. フォルダ選択フレーム
        folder_frame = ttk.LabelFrame(self, text="1. 作業フォルダ", padding=10)
        folder_frame.pack(fill="x", **pad_opts)

        folder_entry = ttk.Entry(folder_frame, textvariable=self.current_folder, state="readonly")
        folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        btn_browse = ttk.Button(folder_frame, text="フォルダ選択...", command=self.choose_folder)
        btn_browse.pack(side="right")

        # 2. FLACファイル選択フレーム
        file_frame = ttk.LabelFrame(self, text="2. 変換するFLACファイルを選択", padding=10)
        file_frame.pack(fill="x", **pad_opts)

        box = ttk.Frame(file_frame)
        box.pack(fill="x", pady=2)
        ttk.Label(box, text="対象FLAC:", width=10).pack(side="left")
        self.cb_file = ttk.Combobox(box, textvariable=self.selected_file_name, state="readonly")
        self.cb_file.pack(side="left", fill="x", expand=True)
        self.cb_file.bind("<<ComboboxSelected>>", lambda e: self.on_file_selected())

        # 3. 出力ファイル名設定
        out_frame = ttk.LabelFrame(self, text="3. 出力設定 (同フォルダに保存されます)", padding=10)
        out_frame.pack(fill="x", **pad_opts)

        ttk.Label(out_frame, text="出力WAV名:").pack(side="left", padx=(0, 6))
        out_entry = ttk.Entry(out_frame, textvariable=self.output_filename)
        out_entry.pack(side="left", fill="x", expand=True)

        # 実行ボタン & ステータス
        action_frame = ttk.Frame(self, padding=10)
        action_frame.pack(fill="x", **pad_opts)

        self.btn_run = ttk.Button(action_frame, text="WAVに変換する", command=self.run_convert)
        self.btn_run.pack(fill="x", ipady=5)

        self.status_label = ttk.Label(self, text="準備完了", foreground="gray")
        self.status_label.pack(side="bottom", anchor="w", padx=12, pady=(0, 8))

    def choose_folder(self):
        initial = self.current_folder.get()
        if not os.path.isdir(initial):
            initial = os.getcwd()

        selected = filedialog.askdirectory(initialdir=initial, title="FLACがあるフォルダを選択")
        if selected:
            self.current_folder.set(selected)
            self.config_data["last_folder"] = selected
            save_config(self.config_data)
            self.refresh_flac_list()

    def refresh_flac_list(self):
        folder = self.current_folder.get()
        self.file_map.clear()

        if not os.path.exists(folder):
            self.cb_file["values"] = []
            self.selected_file_name.set("フォルダが存在しません")
            return

        try:
            entries = sorted(os.listdir(folder))
            flac_files = [f for f in entries if f.lower().endswith(".flac")]
        except Exception as e:
            messagebox.showerror("エラー", f"フォルダの読み込みに失敗しました:\n{e}")
            return

        for fname in flac_files:
            self.file_map[fname] = os.path.join(folder, fname)

        if flac_files:
            self.cb_file["values"] = flac_files
            self.selected_file_name.set(flac_files[0])
            self.on_file_selected()
            self.status_label.config(text=f"{len(flac_files)} 個のFLACファイルが見つかりました", foreground="black")
        else:
            self.cb_file["values"] = []
            self.selected_file_name.set("FLACファイルが見つかりません")
            self.output_filename.set("")
            self.status_label.config(text="FLACファイルが見つかりません", foreground="red")

    def on_file_selected(self):
        sel = self.selected_file_name.get()
        if sel and sel in self.file_map:
            base_name, _ = os.path.splitext(sel)
            self.output_filename.set(f"{base_name}.wav")

    def run_convert(self):
        sel_name = self.selected_file_name.get()
        in_file = self.file_map.get(sel_name)

        if not in_file or not os.path.isfile(in_file):
            messagebox.showwarning("入力エラー", "有効なFLACファイルを選択してください。")
            return

        out_name = self.output_filename.get().strip()
        if not out_name:
            messagebox.showwarning("入力エラー", "出力ファイル名を入力してください。")
            return

        if not out_name.lower().endswith(".wav"):
            out_name += ".wav"
            self.output_filename.set(out_name)

        out_path = os.path.join(self.current_folder.get(), out_name)

        if os.path.exists(out_path):
            if not messagebox.askyesno("上書き確認", f"'{out_name}' は既に存在します。\n上書きしますか？"):
                return

        self.btn_run.config(state="disabled")
        self.status_label.config(text="変換中...", foreground="blue")
        self.update_idletasks()

        cmd = [
            "ffmpeg",
            "-y",
            "-i", in_file,
            out_path
        ]

        try:
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                startupinfo=startupinfo
            )

            if result.returncode != 0:
                err_msg = result.stderr.strip()
                if not err_msg:
                    err_msg = result.stdout.strip()
                raise RuntimeError(err_msg or "ffmpeg execution failed")

            self.status_label.config(text=f"変換完了: {out_name}", foreground="green")
            messagebox.showinfo("完了", f"変換が完了しました！\n\n出力先:\n{out_path}")

        except FileNotFoundError:
            self.status_label.config(text="エラー: ffmpegが見つかりません", foreground="red")
            messagebox.showerror(
                "ffmpegが見つかりません",
                "ffmpegコマンドが実行できませんでした。\nシステム環境変数 PATH に ffmpeg が通っているか確認してください。"
            )
        except Exception as e:
            self.status_label.config(text="変換失敗", foreground="red")
            messagebox.showerror("エラー", f"変換に失敗しました:\n\n{e}")
        finally:
            self.btn_run.config(state="normal")

if __name__ == "__main__":
    app = FlacToWavApp()
    app.mainloop()
