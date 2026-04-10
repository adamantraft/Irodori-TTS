#!/usr/bin/env python3
"""
Irodori-TTS Web UI (Streamlit)
話者管理・バッチ生成・音声再生をブラウザから操作できます。
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import subprocess
from pathlib import Path

import streamlit as st

from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    save_wav,
)

# -----------------------------------------------------------------------
# 定数
# -----------------------------------------------------------------------
SPEAKERS_FILE = Path("speakers.json")
OUTPUTS_DIR = Path("outputs")
DEFAULT_MODEL = "Aratako/Irodori-TTS-500M-v2"
VOICEDESIGN_MODEL = "Aratako/Irodori-TTS-500M-v2-VoiceDesign"
DEVICE = "cuda"
# 環境変数 TTS_PRECISION で精度を指定可能 (デフォルト: bf16)
# 例: SET TTS_PRECISION=fp32 (Windows) / export TTS_PRECISION=fp32 (Linux/Mac)
PRECISION = os.getenv("TTS_PRECISION", "bf16")

# -----------------------------------------------------------------------
# ユーティリティ
# -----------------------------------------------------------------------

def load_speakers() -> dict:
    if SPEAKERS_FILE.exists():
        with open(SPEAKERS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_speakers(speakers: dict) -> None:
    with open(SPEAKERS_FILE, "w", encoding="utf-8") as f:
        json.dump(speakers, f, ensure_ascii=False, indent=2)


@st.cache_data(show_spinner=False)
def resolve_checkpoint_path(hf_repo: str) -> str:
    return hf_hub_download(repo_id=hf_repo, filename="model.safetensors")


@st.cache_resource(show_spinner="モデルを読み込み中...")
def get_runtime(checkpoint_path: str) -> InferenceRuntime:
    key = RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=DEVICE,
        model_precision=PRECISION,
        codec_device=DEVICE,
        codec_precision=PRECISION,
    )
    return InferenceRuntime.from_key(key)


def synthesize_one(
    text: str,
    caption: str | None,
    ref_wav: str | None,
    no_ref: bool,
    seed: int | None,
    hf_repo: str,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
) -> tuple[int, bytes]:
    """音声を1つ生成してWAVバイト列を返す。"""
    ckpt = resolve_checkpoint_path(hf_repo)
    runtime = get_runtime(ckpt)
    req = SamplingRequest(
        text=text,
        caption=caption or None,
        ref_wav=ref_wav or None,
        no_ref=no_ref,
        seed=seed,
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
    )
    result = runtime.synthesize(req, log_fn=None)

    import soundfile as sf
    buf = io.BytesIO()
    audio_np = result.audio.squeeze(0).to(dtype=__import__("torch").float32).numpy()
    sf.write(buf, audio_np, result.sample_rate, format="WAV")
    buf.seek(0)
    return result.used_seed, buf.read()


def concat_wavs_ffmpeg(wav_files: list[Path], output: Path, silence_ms: int) -> None:
    list_file = output.parent / "_concat_list.txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for i, wav in enumerate(wav_files):
                f.write(f"file '{wav.as_posix()}'\n")
                if silence_ms > 0 and i < len(wav_files) - 1:
                    f.write(f"duration {silence_ms / 1000:.3f}\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", str(list_file), "-c", "copy", str(output)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg失敗:\n{r.stderr}")
    finally:
        if list_file.exists():
            list_file.unlink()


# -----------------------------------------------------------------------
# ページ: 話者管理
# -----------------------------------------------------------------------

def page_speakers() -> None:
    st.header("話者管理")
    speakers = load_speakers()

    # --- 話者一覧 ---
    st.subheader("登録済み話者")
    if not speakers:
        st.info("まだ話者が登録されていません。")
    else:
        for name, cfg in list(speakers.items()):
            with st.expander(f"🎙️ {name}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.json(cfg)
                with col2:
                    if st.button("削除", key=f"del_{name}"):
                        del speakers[name]
                        save_speakers(speakers)
                        st.experimental_rerun()

    st.divider()

    # --- 新規登録 ---
    st.subheader("話者を追加・上書き")
    mode = st.radio("モード", ["VoiceDesign（テキストで声を指定）", "参照音声（ボイスクローン）", "参照なし"], horizontal=True)

    new_name = st.text_input("話者名", placeholder="ずんだもん")

    new_cfg: dict = {}

    if mode == "VoiceDesign（テキストで声を指定）":
        new_cfg["hf_checkpoint"] = VOICEDESIGN_MODEL
        caption = st.text_area(
            "キャプション（声のスタイル）",
            placeholder="元気で明るい若い女性の声で、テンポよく話してください。",
        )
        new_cfg["caption"] = caption

        seed_mode = st.radio("Seed", ["ランダム（試聴して決める）", "固定値を入力"], horizontal=True)
        seed_val: int | None = None
        if seed_mode == "固定値を入力":
            seed_val = st.number_input("Seed値", min_value=0, max_value=2**31, value=12345, step=1)
            new_cfg["seed"] = int(seed_val)

        # 試聴
        trial_text = st.text_input("試聴テキスト", value="こんにちは、テスト音声です。")
        with st.expander("生成パラメータ（謎音声が出る場合に調整）"):
            trial_cfg_text = st.slider("CFGスケール（テキスト）", 1.0, 10.0, 5.0, step=0.5, key="trial_cfg_text",
                                       help="高いほどテキスト通りの発音になる。謎音声が出る場合は上げてみてください。")
            trial_steps = st.slider("ステップ数", 20, 100, 40, step=10, key="trial_steps",
                                    help="高いほど品質が上がるが遅くなる。")
        if st.button("試聴する"):
            if not caption.strip():
                st.warning("キャプションを入力してください。")
            elif not trial_text.strip():
                st.warning("試聴テキストを入力してください。")
            else:
                with st.spinner("生成中..."):
                    used_seed, wav_bytes = synthesize_one(
                        text=trial_text,
                        caption=caption,
                        ref_wav=None,
                        no_ref=True,
                        seed=seed_val,
                        hf_repo=VOICEDESIGN_MODEL,
                        num_steps=trial_steps,
                        cfg_scale_text=trial_cfg_text,
                    )
                st.audio(wav_bytes, format="audio/wav")
                st.success(f"生成完了！使用seed: `{used_seed}`")
                if seed_mode == "ランダム（試聴して決める）":
                    st.info(f"この声を固定したい場合は seed = `{used_seed}` を設定してください。")
                    if st.button(f"seed {used_seed} を使う"):
                        new_cfg["seed"] = used_seed

    elif mode == "参照音声（ボイスクローン）":
        new_cfg["hf_checkpoint"] = DEFAULT_MODEL
        uploaded = st.file_uploader("参照音声WAVファイル", type=["wav", "mp3", "ogg"])
        if uploaded:
            ref_path = Path("uploads") / uploaded.name
            ref_path.parent.mkdir(exist_ok=True)
            ref_path.write_bytes(uploaded.read())
            new_cfg["ref_wav"] = str(ref_path)
            st.success(f"アップロード完了: {ref_path}")

            trial_text = st.text_input("試聴テキスト", value="こんにちは、テスト音声です。")
            if st.button("試聴する"):
                with st.spinner("生成中..."):
                    used_seed, wav_bytes = synthesize_one(
                        text=trial_text,
                        caption=None,
                        ref_wav=str(ref_path),
                        no_ref=False,
                        seed=None,
                        hf_repo=DEFAULT_MODEL,
                    )
                st.audio(wav_bytes, format="audio/wav")

    else:  # 参照なし
        new_cfg["hf_checkpoint"] = DEFAULT_MODEL
        new_cfg["no_ref"] = True

    st.divider()
    if st.button("💾 保存", type="primary"):
        if not new_name.strip():
            st.warning("話者名を入力してください。")
        else:
            speakers[new_name.strip()] = new_cfg
            save_speakers(speakers)
            st.success(f"「{new_name}」を保存しました！")
            st.experimental_rerun()


# -----------------------------------------------------------------------
# ページ: バッチ生成
# -----------------------------------------------------------------------

def page_batch() -> None:
    st.header("バッチ生成")
    speakers = load_speakers()

    if not speakers:
        st.warning("先に「話者管理」タブで話者を登録してください。")
        return

    st.subheader("台本入力")
    st.caption("形式: `話者名,セリフ`　（#から始まる行はコメント）")

    # 登録済み話者のサンプルを自動生成
    sample_lines = "\n".join(
        f"{name},こんにちは、{name}です。"
        for name in list(speakers.keys())[:2]
    )
    script_text = st.text_area(
        "台本",
        value=sample_lines,
        height=200,
        placeholder="ずんだもん,今日は何の話をしようか？\n四国めたん,天気の話はどうかな。",
    )

    silence_ms = st.slider("発話間の無音（ミリ秒）", 0, 1000, 300, step=50)

    with st.expander("生成パラメータ（謎音声が出る場合に調整）"):
        batch_cfg_text = st.slider("CFGスケール（テキスト）", 1.0, 10.0, 5.0, step=0.5, key="batch_cfg_text",
                                   help="高いほどテキスト通りの発音になる。謎音声が出る場合は上げてみてください。")
        batch_steps = st.slider("ステップ数", 20, 100, 40, step=10, key="batch_steps",
                                help="高いほど品質が上がるが遅くなる。")

    col1, col2 = st.columns(2)
    with col1:
        output_name = st.text_input("出力ファイル名", value="output_batch.wav")
    with col2:
        keep_parts = st.checkbox("個別ファイルも保存する")

    if st.button("🎙️ 生成開始", type="primary"):
        # 台本パース
        lines: list[tuple[str, str]] = []
        errors: list[str] = []
        for i, row_text in enumerate(script_text.splitlines(), start=1):
            row_text = row_text.strip()
            if not row_text or row_text.startswith("#"):
                continue
            parts = row_text.split(",", 1)
            if len(parts) < 2:
                errors.append(f"{i}行目: カンマがありません → {row_text}")
                continue
            sp, txt = parts[0].strip(), parts[1].strip()
            if sp not in speakers:
                errors.append(f"{i}行目: 未登録の話者「{sp}」")
                continue
            if not txt:
                errors.append(f"{i}行目: セリフが空です")
                continue
            lines.append((sp, txt))

        if errors:
            for e in errors:
                st.error(e)
            return

        if not lines:
            st.warning("有効な行がありません。")
            return

        OUTPUTS_DIR.mkdir(exist_ok=True)
        parts_dir = OUTPUTS_DIR / (Path(output_name).stem + "_parts")
        if keep_parts:
            parts_dir.mkdir(exist_ok=True)

        progress = st.progress(0, text="準備中...")
        status = st.empty()
        part_files: list[Path] = []

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            for idx, (speaker, text) in enumerate(lines):
                pct = int(idx / len(lines) * 100)
                progress.progress(pct, text=f"[{idx+1}/{len(lines)}] {speaker}: {text[:20]}...")
                status.info(f"生成中: {speaker} 「{text[:30]}」")

                sp_cfg = speakers[speaker]
                ref_wav = sp_cfg.get("ref_wav")
                caption = sp_cfg.get("caption")
                seed = sp_cfg.get("seed")
                no_ref = sp_cfg.get("no_ref", False) or (ref_wav is None and caption is not None)
                hf_repo = sp_cfg.get("hf_checkpoint", DEFAULT_MODEL)

                used_seed, wav_bytes = synthesize_one(
                    text=text,
                    caption=caption,
                    ref_wav=ref_wav,
                    no_ref=bool(no_ref),
                    seed=seed,
                    hf_repo=hf_repo,
                    num_steps=batch_steps,
                    cfg_scale_text=batch_cfg_text,
                )

                part_path = tmp_path / f"part_{idx+1:04d}_{speaker}.wav"
                part_path.write_bytes(wav_bytes)
                part_files.append(part_path)

                if keep_parts:
                    (parts_dir / part_path.name).write_bytes(wav_bytes)

            progress.progress(100, text="結合中...")
            status.info("ffmpegで結合中...")

            output_path = OUTPUTS_DIR / output_name
            concat_wavs_ffmpeg(part_files, output_path, silence_ms=silence_ms)

        progress.empty()
        status.empty()
        st.success(f"完了！ → `{output_path}`")
        st.audio(str(output_path), format="audio/wav")
        with open(output_path, "rb") as f:
            st.download_button(
                "ダウンロード",
                data=f.read(),
                file_name=output_name,
                mime="audio/wav",
            )
        if keep_parts:
            st.info(f"個別ファイル保存先: `{parts_dir}`")


# -----------------------------------------------------------------------
# ページ: 生成履歴
# -----------------------------------------------------------------------

def page_history() -> None:
    st.header("生成履歴")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    wav_files = sorted(OUTPUTS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not wav_files:
        st.info("まだ生成されたファイルがありません。")
        return

    for wav in wav_files:
        with st.expander(wav.name):
            st.audio(str(wav), format="audio/wav")
            with open(wav, "rb") as f:
                st.download_button(
                    "ダウンロード",
                    data=f.read(),
                    file_name=wav.name,
                    mime="audio/wav",
                    key=f"dl_{wav.name}",
                )


# -----------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Irodori-TTS UI",
        page_icon="🎙️",
        layout="wide",
    )
    st.title("🎙️ Irodori-TTS Web UI")

    tab1, tab2, tab3 = st.tabs(["話者管理", "バッチ生成", "生成履歴"])
    with tab1:
        page_speakers()
    with tab2:
        page_batch()
    with tab3:
        page_history()


if __name__ == "__main__":
    main()
