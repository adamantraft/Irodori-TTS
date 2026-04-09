#!/usr/bin/env python3
"""
バッチTTS生成スクリプト
CSVファイルから台本を読み込み、音声を生成して1つのファイルに結合します。

CSV形式:
  話者名,セリフ
  ずんだもん,今日は何の話をしようか？
  四国めたん,そうだね、天気の話はどうかな。

話者設定ファイル (speakers.json) の形式は3種類あります:

  ① 参照音声でボイスクローン（ベースモデル用）:
    "キャラ名": {
      "ref_wav": "path/to/voice.wav"
    }

  ② VoiceDesign（テキストで声を指定）:
    "キャラ名": {
      "caption": "元気で明るい若い女性の声で話してください。",
      "seed": 12345
    }
    ※ seed を固定すると毎回同じ声になります。省略するとランダム。

  ③ 参照音声なし（声質はランダム）:
    "キャラ名": { "no_ref": true }

  話者ごとに異なるモデルを使いたい場合は hf_checkpoint を指定:
    "キャラ名": {
      "caption": "...",
      "seed": 42,
      "hf_checkpoint": "Aratako/Irodori-TTS-500M-v2-VoiceDesign"
    }
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    save_wav,
)

# ロード済みランタイムのキャッシュ（モデルごとに1つ保持）
_runtime_cache: dict[str, InferenceRuntime] = {}


def load_speakers(speakers_file: str) -> dict:
    path = Path(speakers_file)
    if not path.exists():
        raise FileNotFoundError(f"話者設定ファイルが見つかりません: {speakers_file}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_script(csv_file: str) -> list[tuple[str, str]]:
    lines = []
    path = Path(csv_file)
    if not path.exists():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_file}")
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#"):
                continue  # 空行・コメント行スキップ
            if len(row) < 2:
                print(f"[警告] {i}行目: カラムが不足しています。スキップします: {row}")
                continue
            speaker = row[0].strip()
            text = row[1].strip()
            if not speaker or not text:
                continue
            lines.append((speaker, text))
    return lines


def resolve_checkpoint(checkpoint: str | None, hf_checkpoint: str | None) -> str:
    if checkpoint:
        p = Path(checkpoint)
        if not p.is_file():
            raise FileNotFoundError(f"チェックポイントが見つかりません: {p}")
        return str(p)
    repo_id = (hf_checkpoint or "").strip()
    if not repo_id:
        raise ValueError("checkpoint か hf_checkpoint のどちらかを指定してください。")
    print(f"[checkpoint] HuggingFaceからダウンロード中: {repo_id}", flush=True)
    return hf_hub_download(repo_id=repo_id, filename="model.safetensors")


def get_runtime(checkpoint_path: str, args: argparse.Namespace) -> InferenceRuntime:
    """チェックポイントパスをキーにランタイムをキャッシュして返す。"""
    if checkpoint_path not in _runtime_cache:
        print(f"[モデル] 読み込み中: {Path(checkpoint_path).name}", flush=True)
        key = RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=args.model_device,
            model_precision=args.model_precision,
            codec_device=args.codec_device,
            codec_precision=args.codec_precision,
        )
        _runtime_cache[checkpoint_path] = InferenceRuntime.from_key(key)
        print(f"[モデル] 読み込み完了: {Path(checkpoint_path).name}", flush=True)
    return _runtime_cache[checkpoint_path]


def concat_wavs_ffmpeg(wav_files: list[Path], output: Path, silence_ms: int) -> None:
    """ffmpegで複数のWAVを結合する。"""
    list_file = output.parent / "_concat_list.txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for i, wav in enumerate(wav_files):
                f.write(f"file '{wav.as_posix()}'\n")
                # 最後のファイル以外に無音を挟む
                if silence_ms > 0 and i < len(wav_files) - 1:
                    f.write(f"duration {silence_ms / 1000:.3f}\n")

        # シンプルなconcatフィルタで結合
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("[ffmpeg stderr]", result.stderr)
            raise RuntimeError("ffmpegによる結合に失敗しました。")
    finally:
        if list_file.exists():
            list_file.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSVの台本からバッチTTS生成して1つのWAVに結合する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", help="ローカルチェックポイントパス (.safetensors / .pt)")
    ckpt_group.add_argument("--hf-checkpoint", help="HuggingFaceリポジトリID")
    parser.add_argument("--script", required=True, help="台本CSVファイルのパス")
    parser.add_argument("--speakers", required=True, help="話者設定JSONファイルのパス")
    parser.add_argument("--output", default="output_batch.wav", help="出力WAVファイルパス (デフォルト: output_batch.wav)")
    parser.add_argument("--work-dir", default=None, help="一時WAVファイルの保存先ディレクトリ (省略時は一時ディレクトリ)")
    parser.add_argument("--keep-parts", action="store_true", help="結合後も個別WAVファイルを残す")
    parser.add_argument("--silence-ms", type=int, default=300, help="発話間に挟む無音時間（ミリ秒、デフォルト: 300）")
    parser.add_argument("--model-device", default=default_runtime_device())
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-device", default=default_runtime_device())
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    args = parser.parse_args()

    # 台本・話者設定を読み込む
    script = load_script(args.script)
    speakers = load_speakers(args.speakers)

    if not script:
        print("台本が空です。終了します。")
        sys.exit(1)

    # 台本内の話者が全員設定されているか確認
    missing = {sp for sp, _ in script if sp not in speakers}
    if missing:
        print(f"[エラー] 話者設定が見つかりません: {missing}")
        print(f"speakers.jsonに以下を追加してください: {list(missing)}")
        sys.exit(1)

    # デフォルトチェックポイントを解決（話者ごとに上書き可能）
    default_checkpoint = resolve_checkpoint(args.checkpoint, args.hf_checkpoint)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_temp = args.work_dir is None
    if use_temp:
        tmp_dir_obj = tempfile.TemporaryDirectory()
        work_dir = Path(tmp_dir_obj.name)
    else:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir_obj = None

    try:
        part_files: list[Path] = []
        total = len(script)

        for idx, (speaker, text) in enumerate(script, start=1):
            sp_cfg = speakers[speaker]
            ref_wav = sp_cfg.get("ref_wav")
            caption = sp_cfg.get("caption")
            seed = sp_cfg.get("seed")  # 固定seedで毎回同じ声になる
            no_ref = sp_cfg.get("no_ref", False) or (ref_wav is None and caption is not None)

            # 話者ごとにモデルを切り替える（指定なければデフォルト）
            sp_checkpoint = resolve_checkpoint(
                sp_cfg.get("checkpoint"),
                sp_cfg.get("hf_checkpoint"),
            ) if (sp_cfg.get("checkpoint") or sp_cfg.get("hf_checkpoint")) else default_checkpoint

            runtime = get_runtime(sp_checkpoint, args)

            part_path = work_dir / f"part_{idx:04d}_{speaker}.wav"
            print(f"[{idx}/{total}] {speaker}: {text[:30]}{'...' if len(text) > 30 else ''}", flush=True)

            req = SamplingRequest(
                text=text,
                caption=caption,
                ref_wav=ref_wav,
                no_ref=bool(no_ref) if ref_wav is None else False,
                seed=seed,
                num_steps=args.num_steps,
                cfg_scale_text=args.cfg_scale_text,
                cfg_scale_speaker=args.cfg_scale_speaker,
            )

            result = runtime.synthesize(req, log_fn=None)
            save_wav(part_path, result.audio, result.sample_rate)
            part_files.append(part_path)
            print(f"  -> 生成完了 (seed={result.used_seed}): {part_path.name}", flush=True)

        # ffmpegで結合
        print(f"\n[結合] {len(part_files)}ファイルを結合中...", flush=True)
        concat_wavs_ffmpeg(part_files, output_path, silence_ms=args.silence_ms)
        print(f"[完了] 出力: {output_path}", flush=True)

        if args.keep_parts and use_temp:
            # 一時ディレクトリ使用時にkeep-partsが指定された場合、出力先に移動
            parts_dir = output_path.parent / (output_path.stem + "_parts")
            parts_dir.mkdir(exist_ok=True)
            for p in part_files:
                dest = parts_dir / p.name
                p.rename(dest)
            print(f"[保存] 個別ファイル: {parts_dir}", flush=True)

    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
