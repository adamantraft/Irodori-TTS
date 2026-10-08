#!/usr/bin/env python3
"""
どのフォルダからでも呼べる音声生成CLI（Claude Codeのスキル irodori-tts 用）

WebUIと同じ speakers.json・参照音声の保存庫を使う。パスはこのファイルの場所を基準に
解決するので、カレントディレクトリはどこでもよい。結果はJSONで標準出力、進捗は標準エラー。

  python tts_cli.py list
  python tts_cli.py say --speaker 28歳ギャル --text "こんにちは" -o hello.wav
  python tts_cli.py say --caption "落ち着いた老人の男性の声。" --text "..." -o old.wav
  python tts_cli.py batch --script daihon.csv -o out.wav --parts-dir parts

台本CSVは WebUI のバッチ生成と同じ `話者名,セリフ` 形式（#から始まる行はコメント）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
SPEAKERS_FILE = REPO_DIR / "speakers.json"
# 以下の既定値は webui.py と揃えてある
VOICES_DIR = Path(os.getenv("TTS_VOICES_DIR", r"H:\マイドライブ\Projects\IrodoriVoices"))
DEFAULT_MODEL = os.getenv("TTS_DEFAULT_MODEL", "Aratako/Irodori-TTS-v4-Large")
LEGACY_MODEL = "Aratako/Irodori-TTS-v4.1-Small"
PRECISION = os.getenv("TTS_PRECISION", "bf16")

# 1回の生成は最大30秒。これを超える長さのセリフは文の切れ目で分割して生成する。
MAX_CHARS = 140
# 分割した文同士の間（ミリ秒）
SENTENCE_GAP_MS = 150
# 空きVRAMがこれ未満なら警告する（WebUIやComfyUIがモデルを載せたままのとき）
LOW_VRAM_MB = 6000


# 結果のJSONだけを書き出す本来の標準出力（main で差し替える）
_RESULT_OUT = sys.stdout


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def emit(result: dict) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2), file=_RESULT_OUT, flush=True)


def fail(msg: str, **extra) -> None:
    emit({"ok": False, "error": msg, **extra})
    sys.exit(1)


def load_speakers() -> dict:
    if not SPEAKERS_FILE.exists():
        fail(f"話者設定ファイルが見つかりません: {SPEAKERS_FILE}")
    with open(SPEAKERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def resolve_ref(ref: str | None) -> str | None:
    """speakers.json の ref_wav を実ファイルのパスに解決する。ファイル名だけなら保存庫から探す。"""
    if not ref:
        return None
    p = Path(ref)
    if p.is_absolute():
        return str(p)
    if (REPO_DIR / p).exists():
        return str(REPO_DIR / p)
    return str(VOICES_DIR / ref)


def speaker_info(name: str, cfg: dict) -> dict:
    ref = resolve_ref(cfg.get("ref_wav"))
    info = {
        "name": name,
        "caption": cfg.get("caption"),
        "model": cfg.get("hf_checkpoint", LEGACY_MODEL),
        "has_ref": ref is not None,
        # 参照音声かseedがあれば毎回同じ声になる
        "stable": ref is not None or cfg.get("seed") is not None,
    }
    # character: この声の持ち主のキャラ（キャラシート名）。note: 声の覚え書き。どちらも生成には使わない
    for key in ("character", "note"):
        if cfg.get(key):
            info[key] = cfg[key]
    if ref is not None and not Path(ref).exists():
        info["problem"] = f"参照音声が見つかりません: {ref}"
    return info


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """長いセリフを文の切れ目で limit 文字以内のかたまりに分ける。"""
    text = text.strip()
    if len(text) <= limit:
        return [text]
    sentences = [s for s in re.split(r"(?<=[。！？!?\n])", text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) > limit:
            chunks.append(current.strip())
            current = ""
        current += s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_script(path: Path) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.reader(f), start=1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                fail(f"{path.name} {i}行目: `話者名,セリフ` の形になっていません: {row}")
            # クォートなしでセリフにカンマが入っている場合は元に戻す
            speaker, text = row[0].strip(), ",".join(row[1:]).strip()
            if speaker and text:
                lines.append((speaker, text))
    return lines


def free_vram_mb() -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free / 1024 / 1024)


def generate(jobs: list[dict], args: argparse.Namespace) -> dict:
    """jobs（speaker, text, cfg）を生成して1つのWAVに結合し、結果の辞書を返す。"""
    import torch

    from irodori_tts.inference_runtime import (
        RuntimeKey,
        SamplingRequest,
        default_runtime_device,
        download_hf_checkpoint,
        get_cached_runtime,
        save_wav,
    )

    warnings: list[str] = []
    vram = free_vram_mb()
    if vram is not None and vram < LOW_VRAM_MB:
        warnings.append(
            f"空きVRAMが{vram}MBしかありません。WebUIやComfyUIがモデルを載せたままだと、"
            "生成が極端に遅くなるか失敗します。"
        )
        log(f"[警告] {warnings[-1]}")

    device = default_runtime_device()
    for job in jobs:
        job["model"] = args.model or job["cfg"].get("hf_checkpoint", LEGACY_MODEL)
        job["chunks"] = split_text(job["text"])
        if any(len(c) > MAX_CHARS for c in job["chunks"]):
            warnings.append(
                f"{job['index']}行目: 句点のない{MAX_CHARS}字超の文があります。30秒に収まらず末尾が詰まる可能性があります。"
            )

    # モデルの再読み込みを減らすため、同じモデルの行をまとめて生成する（結合は台本順）
    order = sorted(range(len(jobs)), key=lambda i: jobs[i]["model"])
    sample_rate = None
    for done, i in enumerate(order, start=1):
        job = jobs[i]
        cfg = job["cfg"]
        ref_wav = resolve_ref(cfg.get("ref_wav"))
        caption = cfg.get("caption") or None
        seed = args.seed if args.seed is not None else cfg.get("seed")

        key = RuntimeKey(
            checkpoint=download_hf_checkpoint(job["model"]),
            model_device=device,
            model_precision=PRECISION,
            codec_device=device,
            codec_precision=PRECISION,
        )
        # LargeとSmallはVRAMに同時に載らない。キーが変わると旧モデルは解放される
        runtime, reloaded = get_cached_runtime(key)
        if reloaded:
            log(f"[モデル] 読み込み完了: {job['model']}")

        log(f"[{done}/{len(jobs)}] {job['speaker']}: {job['text'][:30]}")
        pieces: list[torch.Tensor] = []
        for chunk in job["chunks"]:
            result = runtime.synthesize(
                SamplingRequest(
                    text=chunk,
                    caption=caption,
                    ref_wav=ref_wav,
                    no_ref=ref_wav is None,
                    seed=seed,
                    num_steps=args.steps,
                    cfg_scale_text=args.cfg_text,
                ),
                log_fn=None,
            )
            # 分割した2文目以降も同じ声になるよう、最初に使われたseedを引き継ぐ
            seed = result.used_seed
            sample_rate = result.sample_rate
            pieces.append(result.audio.detach().to(device="cpu", dtype=torch.float32))
        job["seed"] = seed
        job["audio"] = join_audio(pieces, sample_rate, SENTENCE_GAP_MS)

    output = Path(args.output).resolve()
    parts_dir = Path(args.parts_dir).resolve() if args.parts_dir else None
    lines = []
    cursor = 0.0
    for n, job in enumerate(jobs, start=1):
        seconds = job["audio"].shape[-1] / sample_rate
        line = {
            "index": job["index"],
            "speaker": job["speaker"],
            "voice": job["voice"],
            "text": job["text"],
            "model": job["model"],
            "seed": job["seed"],
            "start": round(cursor, 3),
            "end": round(cursor + seconds, 3),
        }
        if parts_dir is not None:
            part = parts_dir / f"{n:04d}_{safe_name(job['speaker'])}.wav"
            save_wav(part, job["audio"], sample_rate)
            line["file"] = str(part)
        lines.append(line)
        cursor += seconds + (args.silence_ms / 1000 if n < len(jobs) else 0)

    save_wav(output, join_audio([j["audio"] for j in jobs], sample_rate, args.silence_ms), sample_rate)
    log(f"[完了] {output}")
    return {
        "ok": True,
        "output": str(output),
        "seconds": round(cursor, 3),
        "sample_rate": sample_rate,
        "lines": lines,
        "warnings": warnings,
    }


def join_audio(pieces: list, sample_rate: int, gap_ms: int):
    import torch

    if len(pieces) == 1:
        return pieces[0]
    gap = torch.zeros(pieces[0].shape[0], int(sample_rate * gap_ms / 1000))
    joined = []
    for i, piece in enumerate(pieces):
        if i:
            joined.append(gap)
        joined.append(piece)
    return torch.cat(joined, dim=-1)


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]', "_", name)


def lookup(speakers: dict, name: str) -> dict:
    if name not in speakers:
        fail(f"未登録の話者です: {name}", speakers=list(speakers))
    info = speaker_info(name, speakers[name])
    if "problem" in info:
        fail(f"話者「{name}」: {info['problem']}")
    return speakers[name]


def cmd_list(args: argparse.Namespace) -> None:
    speakers = load_speakers()
    emit({"ok": True, "speakers": [speaker_info(n, c) for n, c in speakers.items()]})


def cmd_say(args: argparse.Namespace) -> None:
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8-sig").strip()
    else:
        text = (args.text or "").strip()
    if not text:
        fail("--text か --text-file でセリフを指定してください。")

    if args.speaker:
        cfg = dict(lookup(load_speakers(), args.speaker))
        if args.caption:
            cfg["caption"] = args.caption
        name = args.speaker
    elif args.caption:
        # 登録話者を使わず、声の説明文だけでその場で声を作る（ボイスデザイン）
        cfg = {"caption": args.caption, "hf_checkpoint": DEFAULT_MODEL}
        name = "voicedesign"
    else:
        fail("--speaker（登録話者）か --caption（声の説明文）のどちらかを指定してください。")

    jobs = [{"index": 1, "speaker": name, "voice": name, "text": text, "cfg": cfg}]
    emit(generate(jobs, args))


def cmd_batch(args: argparse.Namespace) -> None:
    script = Path(args.script)
    if not script.exists():
        fail(f"台本ファイルが見つかりません: {script}")
    lines = load_script(script)
    if not lines:
        fail("台本に有効な行がありません。")

    # 台本上の役名を登録話者に割り当てる（例: --map A=28歳ギャル）
    mapping: dict[str, str] = {}
    for item in args.map or []:
        if "=" not in item:
            fail(f"--map は `台本の名前=登録話者` の形で指定してください: {item}")
        role, voice = item.split("=", 1)
        mapping[role.strip()] = voice.strip()

    speakers = load_speakers()
    missing = sorted({mapping.get(sp, sp) for sp, _ in lines} - set(speakers))
    if missing:
        fail(f"未登録の話者があります: {missing}（--map 台本の名前=登録話者 で割り当てできます）",
             speakers=list(speakers))

    jobs = []
    for i, (speaker, text) in enumerate(lines, start=1):
        voice = mapping.get(speaker, speaker)
        jobs.append({"index": i, "speaker": speaker, "voice": voice, "text": text,
                     "cfg": lookup(speakers, voice)})
    emit(generate(jobs, args))


def add_generation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--output", required=True, help="出力WAVファイルのパス")
    p.add_argument("--parts-dir", help="1行ごとのWAVも保存するフォルダ")
    p.add_argument("--silence-ms", type=int, default=300, help="行と行の間の無音（ミリ秒、デフォルト: 300）")
    p.add_argument("--model", help="話者に登録したモデルを無視して使うHugging FaceリポジトリID")
    p.add_argument("--seed", type=int, help="seedを手動指定（登録済みのseedより優先）")
    p.add_argument("--steps", type=int, default=40, help="ステップ数（デフォルト: 40）")
    p.add_argument("--cfg-text", type=float, default=5.0,
                   help="CFGスケール（テキスト）。謎音声が出るときは上げる（デフォルト: 5.0）")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    # ライブラリが print する進捗で結果のJSONが壊れないよう、通常の出力は標準エラーに流す
    sys.stdout = sys.stderr

    parser = argparse.ArgumentParser(description="Irodori-TTS 音声生成CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="登録話者の一覧")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("say", help="1つのセリフを生成")
    p.add_argument("--speaker", help="登録話者の名前")
    p.add_argument("--caption", help="声の説明文。--speaker なしならこの説明だけで声を作る")
    p.add_argument("--text", help="セリフ")
    p.add_argument("--text-file", help="セリフを書いたテキストファイル（UTF-8）")
    add_generation_args(p)
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("batch", help="台本CSVから生成して1つのWAVに結合")
    p.add_argument("--script", required=True, help="台本CSV（話者名,セリフ）")
    p.add_argument("--map", action="append", metavar="台本の名前=登録話者",
                   help="台本上の役名を登録話者に割り当てる（複数指定可）")
    add_generation_args(p)
    p.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
