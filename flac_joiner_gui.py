import os
import sys
import json
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flac_joiner_config.json")

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

class FlacJoinerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FLAC 2曲結合ツール (WAV出力)")
        self.geometry("560x420")
        self.resizable(False, False)

        self.config_data = load_config()
        self.current_folder = tk.StringVar(value=self.config_data.get("last_folder", os.getcwd()))

        self.file1_name = tk.StringVar(value="未選択")
        self.file2_name = tk.StringVar(value="未選択")
        self.output_filename = tk.StringVar(value="joined.wav")

        self.file_map = {}  # {表示名: フルパス}

        self.setup_ui()
        self.refresh_flac_list()

    def setup_ui(self):
        pad_opts = {'padx': 12, 'pady': 6}

        # フォルダ選択フレーム
        folder_frame = ttk.LabelFrame(self, text="作業フォルダ", padding=10)
        folder_frame.pack(fill="x", **pad_opts)

        folder_entry = ttk.Entry(folder_frame, textvariable=self.current_folder, state="readonly")
        folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        btn_browse = ttk.Button(folder_frame, text="フォルダ選択...", command=self.choose_folder)
        btn_browse.pack(side="right")

        # ファイル選択フレーム
        files_frame = ttk.LabelFrame(self, text="結合するFLACファイル (2つ)", padding=10)
        files_frame.pack(fill="x", **pad_opts)

        # 1つ目
        f1_box = ttk.Frame(files_frame)
        f1_box.pack(fill="x", pady=4)
        ttk.Label(f1_box, text="1曲目 (前):", width=12).pack(side="left")
        self.cb_file1 = ttk.Combobox(f1_box, textvariable=self.file1_name, state="readonly")
        self.cb_file1.pack(side="left", fill="x", expand=True)

        # 2つ目
        f2_box = ttk.Frame(files_frame)
        f2_box.pack(fill="x", pady=4)
        ttk.Label(f2_box, text="2曲目 (後):", width=12).pack(side="left")
        self.cb_file2 = ttk.Combobox(f2_box, textvariable=self.file2_name, state="readonly")
        self.cb_file2.pack(side="left", fill="x", expand=True)

        self.cb_file1.bind("<<ComboboxSelected>>", lambda e: self.update_default_output_name())
        self.cb_file2.bind("<<ComboboxSelected>>", lambda e: self.update_default_output_name())

        # 順番入れ替えボタン
        swap_box = ttk.Frame(files_frame)
        swap_box.pack(fill="x", pady=(4, 0))
        btn_swap = ttk.Button(swap_box, text="↑↓ 前後を入れ替える", command=self.swap_files)
        btn_swap.pack(side="right")

        # 出力ファイル名設定
        out_frame = ttk.LabelFrame(self, text="出力設定 (同フォルダに保存されます)", padding=10)
        out_frame.pack(fill="x", **pad_opts)

        ttk.Label(out_frame, text="出力ファイル名:").pack(side="left", padx=(0, 6))
        out_entry = ttk.Entry(out_frame, textvariable=self.output_filename)
        out_entry.pack(side="left", fill="x", expand=True)

        # 実行ボタン & ステータス
        action_frame = ttk.Frame(self, padding=10)
        action_frame.pack(fill="x", **pad_opts)

        self.btn_run = ttk.Button(action_frame, text="FLACを結合してWAVを作成", command=self.run_join)
        self.btn_run.pack(fill="x", ipady=5)

        self.status_label = ttk.Label(self, text="準備完了", foreground="gray")
        self.status_label.pack(side="bottom", pady=8)

    def choose_folder(self):
        initial_dir = self.current_folder.get()
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()
        selected = filedialog.askdirectory(initialdir=initial_dir, title="FLACのあるフォルダを選択")
        if selected:
            self.current_folder.set(selected)
            self.config_data["last_folder"] = selected
            save_config(self.config_data)
            self.refresh_flac_list()

    def refresh_flac_list(self):
        folder = self.current_folder.get()
        self.file_map = {}
        if os.path.exists(folder):
            try:
                for entry in sorted(os.listdir(folder)):
                    if entry.lower().endswith(".flac"):
                        full_path = os.path.join(folder, entry)
                        if os.path.isfile(full_path):
                            self.file_map[entry] = full_path
            except Exception as e:
                self.status_label.config(text=f"フォルダ読み込みエラー: {e}", foreground="red")

        file_list = list(self.file_map.keys())
        self.cb_file1["values"] = file_list
        self.cb_file2["values"] = file_list

        if len(file_list) >= 2:
            self.file1_name.set(file_list[0])
            self.file2_name.set(file_list[1])
            self.update_default_output_name()
            self.status_label.config(text=f"{len(file_list)} 個のFLACファイルが見つかりました。", foreground="green")
        elif len(file_list) == 1:
            self.file1_name.set(file_list[0])
            self.file2_name.set("未選択")
            self.status_label.config(text="FLACファイルが1つしかありません (結合には2つ必要です)。", foreground="orange")
        else:
            self.file1_name.set("未選択")
            self.file2_name.set("未選択")
            self.status_label.config(text="選択フォルダにFLACファイルが見つかりません。", foreground="gray")

    def update_default_output_name(self):
        f1 = os.path.splitext(self.file1_name.get())[0]
        f2 = os.path.splitext(self.file2_name.get())[0]
        if f1 and f2 and f1 != "未選択" and f2 != "未選択":
            self.output_filename.set(f"{f1}_{f2}.wav")

    def swap_files(self):
        v1 = self.file1_name.get()
        v2 = self.file2_name.get()
        self.file1_name.set(v2)
        self.file2_name.set(v1)
        self.update_default_output_name()

    def run_join(self):
        f1_key = self.file1_name.get()
        f2_key = self.file2_name.get()

        if f1_key not in self.file_map or f2_key not in self.file_map:
            messagebox.showerror("エラー", "結合するFLACファイルを2つ正しく選択してください。")
            return

        p1 = self.file_map[f1_key]
        p2 = self.file_map[f2_key]

        out_name = self.output_filename.get().strip()
        if not out_name:
            out_name = "joined.wav"
        if not out_name.lower().endswith(".wav"):
            out_name += ".wav"

        out_path = os.path.join(self.current_folder.get(), out_name)

        self.btn_run.config(state="disabled")
        self.status_label.config(text="結合処理中...", foreground="blue")
        self.update()

        try:
            # ffmpeg concat filter を使用してリサンプリング/チャンネル差異も吸収して結合
            cmd = [
                "ffmpeg", "-y",
                "-i", p1,
                "-i", p2,
                "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[outa]",
                "-map", "[outa]",
                out_path
            ]
            
            # Windowsで黒いコマンドプロンプト画面をポップアップさせない設定
            startupinfo = None
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                startupinfo=startupinfo,
                creationflags=creationflags
            )

            if result.returncode == 0:
                self.status_label.config(text=f"出力完了: {out_name}", foreground="green")
                messagebox.showinfo("成功", f"WAVファイルを出力しました！\n\n保存先:\n{out_path}")
            else:
                self.status_label.config(text="ffmpeg エラー", foreground="red")
                messagebox.showerror("エラー", f"結合に失敗しました:\n\n{result.stderr[-500:]}")

        except FileNotFoundError:
            self.status_label.config(text="ffmpegが見つかりません", foreground="red")
            messagebox.showerror(
                "ffmpegエラー",
                "ffmpeg コマンドが見つかりませんでした。\n環境変数PATHに ffmpeg が登録されているか確認してください。"
            )
        except Exception as e:
            self.status_label.config(text="例外エラー発生", foreground="red")
            messagebox.showerror("エラー", f"予期しないエラーが発生しました:\n{e}")
        finally:
            self.btn_run.config(state="normal")


if __name__ == "__main__":
    app = FlacJoinerApp()
    app.mainloop()
