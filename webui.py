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
import time
from pathlib import Path

import streamlit as st

from irodori_tts import inference_runtime
from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_ITEMS
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    download_hf_checkpoint,
    get_cached_runtime,
    save_wav,
)

# -----------------------------------------------------------------------
# 定数
# -----------------------------------------------------------------------
SPEAKERS_FILE = Path("speakers.json")
OUTPUTS_DIR = Path("outputs")
# ボイスデザインで作った参照音声の保存庫（Google Drive同期フォルダ）。環境変数 TTS_VOICES_DIR で変更可能。
# speakers.json にはファイル名だけを記録し、実体はこのフォルダから探す。
VOICES_DIR = Path(os.getenv("TTS_VOICES_DIR", r"H:\マイドライブ\Projects\IrodoriVoices"))
# v4系はテキスト・参照音声・キャプションの3系統を1チェックポイントに統合
# Large: VoiceDesignの再現度・声の類似度が高い / Small: 軽量・高速で漢字の読みがやや正確
MODELS = {
    "v4-Large（高品質）": "Aratako/Irodori-TTS-v4-Large",
    "v4.1-Small（軽量・高速）": "Aratako/Irodori-TTS-v4.1-Small",
}
# 環境変数 TTS_DEFAULT_MODEL で新規話者の既定モデルを指定可能 (デフォルト: v4-Large)
DEFAULT_MODEL = os.getenv("TTS_DEFAULT_MODEL", "Aratako/Irodori-TTS-v4-Large")
# hf_checkpoint 未指定の旧話者データ用（従来の既定モデル）
LEGACY_MODEL = "Aratako/Irodori-TTS-v4.1-Small"
DEVICE = "cuda"
# 環境変数 TTS_PRECISION で精度を指定可能 (デフォルト: bf16)
# 例: SET TTS_PRECISION=fp32 (Windows) / export TTS_PRECISION=fp32 (Linux/Mac)
PRECISION = os.getenv("TTS_PRECISION", "bf16")

# ボイスデザイン用の試聴テキスト。参照音声にも使うので、推奨の約30秒に近い長さにしてある。
# 1回の生成は最大30秒なので、これ以上長くすると末尾が詰まる（目安: 150字前後で25〜28秒）。
TRIAL_TEXTS: dict[str, str] = {
    "ナレーション（落ち着いた説明）": (
        "朝の光がカーテンの隙間から差し込み、部屋の中をゆっくりと照らしていきます。"
        "今日は少し早起きをして、温かいお茶を淹れ、ゆったりとした時間を過ごすことにしました。"
        "窓を開けると、ひんやりとした風が頬をなで、遠くから鳥のさえずりが聞こえてきます。"
        "こんな静かな朝は、久しぶりかもしれません。"
    ),
    "日常会話（明るい雑談）": (
        "ねえ、昨日のドラマ見た？もう最高だったんだけど！最後のシーンなんて、思わず声が出ちゃったよ。"
        "今度一緒に見返そうよ。絶対もう一回泣くと思うから。"
        "あ、そうだ、駅前に新しいカフェができたの知ってる？チーズケーキがすごく美味しいらしいんだ。"
        "今度の休みに行ってみない？"
    ),
    "感情豊か（喜び→驚き→落ち込み）": (
        "やった、合格だ！本当に嬉しい、夢みたい。……えっ、待って、これって本当に私の番号？"
        "見間違いじゃないよね。ああ、よかった。ずっと不安で、昨日の夜は眠れなかったんだから。"
        "……でも、一緒に頑張ってきたあの子の番号が、どこにもないんだ。"
        "なんて声をかけたらいいんだろう。素直に喜べないよ。"
    ),
    "ビジネス（丁寧な案内）": (
        "本日はお忙しい中、ご参加いただき誠にありがとうございます。"
        "これより、新しいサービスの概要とスケジュールについて、順を追ってご説明いたします。"
        "お手元の資料は、全部で三部ございます。"
        "ご不明な点がございましたら、最後にまとめてお伺いいたします。"
        "それでは、最初のページをご覧ください。"
    ),
    "物語（低めの語り）": (
        "むかしむかし、深い山の奥に、一軒の古びた家がありました。"
        "そこには年老いた木こりが一人で暮らしていて、毎晩、囲炉裏の火を見つめながら、遠い昔の話を思い出していたそうです。"
        "ある雪の夜のこと、戸を叩く小さな音がしました。"
        "こんな夜更けに、いったい誰が訪ねてきたのでしょう。"
    ),
    "早口・元気（アナウンス風）": (
        "さあ始まりました、本日のスペシャルステージ！最初に登場するのは、今もっとも注目のあのグループです。"
        "皆さん、大きな拍手でお迎えください！準備はいいですか、それでは行きましょう！"
        "会場の熱気も最高潮です！このあとも豪華なゲストが続々と登場しますので、最後までお見逃しなく！"
    ),
}


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
    return download_hf_checkpoint(hf_repo)


def model_label(hf_repo: str) -> str:
    for label, repo in MODELS.items():
        if repo == hf_repo:
            return label
    return hf_repo


def select_model(label: str, key: str, help: str | None = None) -> str:
    """モデル選択UIを表示し、選ばれたHugging FaceリポジトリIDを返す。"""
    repos = list(MODELS.values())
    index = repos.index(DEFAULT_MODEL) if DEFAULT_MODEL in repos else 0
    return st.selectbox(label, repos, index=index, format_func=model_label, key=key, help=help)


def select_model_override(key: str) -> str | None:
    """話者に登録したモデルを上書きする選択UI。上書きしない場合は None を返す。"""
    per_speaker = "話者の設定を使う"
    choice = st.selectbox(
        "使用モデル",
        [per_speaker, *MODELS.values()],
        format_func=lambda v: v if v == per_speaker else f"{model_label(v)} に切り替え",
        key=key,
        help="切り替えると話者に登録したモデルを無視して生成します。"
             "参照音声のない話者は、登録時と違うモデルでは同じseedでも声が変わります。",
    )
    return None if choice == per_speaker else choice


def speaker_model(cfg: dict, override: str | None = None) -> str:
    return override or cfg.get("hf_checkpoint", LEGACY_MODEL)


def get_runtime(checkpoint_path: str, hf_repo: str) -> InferenceRuntime:
    key = RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=DEVICE,
        model_precision=PRECISION,
        codec_device=DEVICE,
        codec_precision=PRECISION,
    )
    if inference_runtime._RUNTIME_CACHE_KEY == key:
        return get_cached_runtime(key)[0]
    # LargeとSmallはVRAMに同時に載せず、切替時は旧モデルを先に解放する
    clear_cached_runtime()
    with st.spinner(f"モデルを読み込み中... ({model_label(hf_repo)})"):
        return get_cached_runtime(key)[0]


def _append_emoji(text_key: str, emoji: str) -> None:
    st.session_state[text_key] = st.session_state.get(text_key, "") + emoji


def emoji_palette(text_key: str, columns: int = 8) -> None:
    """絵文字ボタンを並べ、押すと text_key のテキスト欄の末尾に絵文字を追加する。"""
    with st.expander("😊 絵文字で話し方を変える（押すと末尾に追加）"):
        st.caption("効かせたい文の前後に置いて試してください。位置は入力欄で自由に動かせます。"
                   "ボタンにカーソルを合わせると説明が出ます。")
        for row_start in range(0, len(EMOJI_PALETTE_ITEMS), columns):
            cols = st.columns(columns)
            for col, item in zip(cols, EMOJI_PALETTE_ITEMS[row_start:row_start + columns]):
                col.button(
                    f"{item.emoji} {item.label}",
                    key=f"emo_{text_key}_{item.emoji}",
                    help=item.description,
                    on_click=_append_emoji,
                    args=(text_key, item.emoji),
                    use_container_width=True,
                )


def resolve_ref(ref: str | None) -> str | None:
    """speakers.json の ref_wav を実ファイルのパスに解決する。ファイル名だけなら保存庫から探す。"""
    if not ref:
        return None
    p = Path(ref)
    if p.is_absolute() or p.exists():
        return str(p)
    return str(VOICES_DIR / ref)


def is_library_ref(ref: str | None) -> bool:
    """保存庫の音声か（ディレクトリ部分を持たないファイル名だけの指定）。"""
    return bool(ref) and Path(ref).name == ref


_ILLEGAL_NAME_CHARS = set('\\/:*?"<>|')


def save_voice_to_library(name: str, wav: bytes) -> str | None:
    """保存庫にwavを書き込み、speakers.json に書くファイル名を返す。失敗時は画面にエラーを出してNone。"""
    if any(ch in _ILLEGAL_NAME_CHARS for ch in name):
        st.error('名前に使えない文字があります: \\ / : * ? " < > |')
        return None
    try:
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        (VOICES_DIR / f"{name}.wav").write_bytes(wav)
    except OSError as e:
        st.error(f"保存庫「{VOICES_DIR}」に書き込めません（Google Driveが起動していますか？）: {e}")
        return None
    return f"{name}.wav"


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
    runtime = get_runtime(ckpt, hf_repo)
    req = SamplingRequest(
        text=text,
        caption=caption or None,
        ref_wav=resolve_ref(ref_wav),
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
            with st.expander(f"🎙️ {name}　[{model_label(speaker_model(cfg))}]"):
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
    model_repo = select_model(
        "モデル",
        key="speaker_model",
        help="Large: VoiceDesignの再現度・声の類似度が高い。Small: 軽量・高速で漢字の読みがやや正確。"
             "同じseedでもモデルが違うと別の声になります。",
    )

    new_cfg: dict = {"hf_checkpoint": model_repo}

    if mode == "VoiceDesign（テキストで声を指定）":
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
        else:
            last_seed = st.session_state.get("vd_last_seed")
            if last_seed is not None:
                if st.checkbox(f"試聴したseed（{last_seed}）で声を固定する", key="vd_fix_seed_cb"):
                    new_cfg["seed"] = last_seed
                    seed_val = last_seed

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
                        hf_repo=model_repo,
                        num_steps=trial_steps,
                        cfg_scale_text=trial_cfg_text,
                    )
                st.audio(wav_bytes, format="audio/wav")
                st.success(f"生成完了！使用seed: `{used_seed}`")
                st.session_state["vd_last_seed"] = used_seed
                if seed_mode == "ランダム（試聴して決める）":
                    st.info(f"この声を固定したい場合は「試聴したseed（{used_seed}）で声を固定する」にチェックを入れてください。")

    elif mode == "参照音声（ボイスクローン）":
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
                        hf_repo=model_repo,
                    )
                st.audio(wav_bytes, format="audio/wav")

    else:  # 参照なし
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
# ページ: ボイスデザイン
# -----------------------------------------------------------------------

def page_voicedesign() -> None:
    st.header("ボイスデザイン")
    st.caption("声の特徴を文章で指定して候補を作り、気に入ったものに名前をつけて保存庫に送ります。"
               "保存した声は、他のタブの「話者」からその名前で使えます。")

    caption = st.text_area(
        "声の特徴（キャプション）", height=80, key="vd_caption",
        placeholder="例: 落ち着いた若い女性の声。少し低めで、柔らかく話す。標準語。",
    )

    preset = st.selectbox("試聴テキスト（選ぶと下の欄に入ります）", list(TRIAL_TEXTS.keys()), key="vd_preset")
    # プリセットを切り替えたときだけ、編集欄を書き換える
    if st.session_state.get("vd_preset_applied") != preset:
        st.session_state["vd_text"] = TRIAL_TEXTS[preset]
        st.session_state["vd_preset_applied"] = preset
    text = st.text_area("試聴テキスト（自由に編集できます）", height=110, key="vd_text")

    col1, col2 = st.columns([1, 2])
    with col1:
        vd_model = select_model(
            "モデル",
            key="vd_model",
            help="Large: キャプションの再現度が高い。Small: 軽量・高速で漢字の読みがやや正確。",
        )
        n_cand = st.number_input("一度に作る候補数", min_value=1, max_value=4, value=3, step=1, key="vd_n")
    with col2:
        with st.expander("生成パラメータ"):
            steps = st.slider("ステップ数", 20, 100, 40, step=10, key="vd_steps")
            cfg_text = st.slider("CFGスケール（テキスト）", 1.0, 10.0, 5.0, step=0.5, key="vd_cfg")

    if st.button("🎲 候補を生成", type="primary", key="vd_go"):
        if not caption.strip():
            st.warning("声の特徴を入力してください。")
        elif not text.strip():
            st.warning("試聴テキストを入力してください。")
        else:
            cands = []
            bar = st.progress(0, text="生成中...")
            for i in range(int(n_cand)):
                bar.progress(int(i / n_cand * 100), text=f"生成中... ({i + 1}/{int(n_cand)})")
                used_seed, wav = synthesize_one(
                    text=text, caption=caption, ref_wav=None, no_ref=True, seed=None,
                    hf_repo=vd_model, num_steps=steps, cfg_scale_text=cfg_text,
                )
                cands.append({"seed": used_seed, "wav": wav, "caption": caption, "model": vd_model})
            bar.empty()
            st.session_state["vd_cands"] = cands

    cands = st.session_state.get("vd_cands", [])
    if cands:
        st.subheader("候補")
        for i, c in enumerate(cands):
            with st.container():
                st.markdown(f"**候補 {i + 1}**　seed: `{c['seed']}`　{model_label(c['model'])}")
                st.audio(c["wav"], format="audio/wav")
                ncol, bcol = st.columns([3, 1])
                with ncol:
                    voice_name = st.text_input("保存する名前", key=f"vd_name_{i}", placeholder="例: ナレーター女性A",
                                               label_visibility="collapsed")
                with bcol:
                    if st.button("💾 保存庫へ", key=f"vd_save_{i}"):
                        name = voice_name.strip()
                        if not name:
                            st.warning("名前を入力してください。")
                        else:
                            speakers = load_speakers()
                            if name in speakers:
                                st.warning(f"「{name}」は既にあります。別の名前にするか、話者管理で削除してください。")
                            else:
                                fname = save_voice_to_library(name, c["wav"])
                                if fname:
                                    speakers[name] = {
                                        "hf_checkpoint": c["model"],
                                        "caption": c["caption"],
                                        "seed": c["seed"],
                                        "ref_wav": fname,
                                    }
                                    save_speakers(speakers)
                                    st.success(f"「{name}」を保存庫に保存しました。他のタブの話者で使えます。")

    st.divider()
    st.subheader("保存庫")
    st.caption(f"保存先: `{VOICES_DIR}`")
    lib = {n: c for n, c in load_speakers().items() if is_library_ref(c.get("ref_wav"))}
    if not lib:
        st.info("まだ保存された声がありません。")
    for n, c in lib.items():
        with st.expander(f"🎙️ {n}　[{model_label(speaker_model(c))}]"):
            st.caption(c.get("caption", ""))
            ref_path = resolve_ref(c["ref_wav"])
            if Path(ref_path).exists():
                st.audio(ref_path, format="audio/wav")
            else:
                st.warning(f"参照音声が見つかりません（Google Driveの同期待ち？）: {ref_path}")


# -----------------------------------------------------------------------
# ページ: 単発生成
# -----------------------------------------------------------------------

def page_single() -> None:
    st.header("単発生成")
    speakers = load_speakers()

    if not speakers:
        st.warning("先に「話者管理」タブで話者を登録してください。")
        return

    name = st.selectbox("話者", list(speakers.keys()), key="single_speaker")
    cfg = speakers[name]
    if cfg.get("caption"):
        st.caption(f"声のスタイル: {cfg['caption']}")
    if cfg.get("ref_wav"):
        st.caption(f"参照音声: {cfg['ref_wav']}")
    if not cfg.get("caption") and not cfg.get("ref_wav"):
        st.caption("参照なし")
    if cfg.get("seed") is None:
        st.caption("Seed未固定: 生成のたびに声が少し変わります。気に入ったら下の「この声を固定」で保存できます。")
    else:
        st.caption(f"Seed固定: {cfg['seed']}")
    st.caption(f"登録モデル: {model_label(speaker_model(cfg))}")
    single_model = speaker_model(cfg, select_model_override("single_model"))

    emoji_palette("single_text")
    text = st.text_area("セリフ", height=120, key="single_text", placeholder="ここに読み上げたい文章を入力")

    with st.expander("生成パラメータ"):
        steps = st.slider("ステップ数", 20, 100, 40, step=10, key="single_steps",
                          help="高いほど品質が上がるが遅くなる。")
        cfg_text = st.slider("CFGスケール（テキスト）", 1.0, 10.0, 5.0, step=0.5, key="single_cfg",
                             help="高いほどテキスト通りの発音になる。謎音声が出る場合は上げてみてください。")
        override_seed = st.checkbox("Seedを手動指定する", key="single_seed_on")
        manual_seed = st.number_input("Seed", min_value=0, max_value=2**31, value=12345, step=1,
                                      key="single_seed_val", disabled=not override_seed)

    if st.button("🎙️ 生成", type="primary", key="single_go"):
        if not text.strip():
            st.warning("セリフを入力してください。")
        else:
            ref_wav = cfg.get("ref_wav")
            caption = cfg.get("caption")
            no_ref = bool(cfg.get("no_ref", False) or (ref_wav is None and caption is not None))
            seed = int(manual_seed) if override_seed else cfg.get("seed")
            with st.spinner("生成中..."):
                used_seed, wav_bytes = synthesize_one(
                    text=text,
                    caption=caption,
                    ref_wav=ref_wav,
                    no_ref=no_ref,
                    seed=seed,
                    hf_repo=single_model,
                    num_steps=steps,
                    cfg_scale_text=cfg_text,
                )
            OUTPUTS_DIR.mkdir(exist_ok=True)
            fname = f"single_{name}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            (OUTPUTS_DIR / fname).write_bytes(wav_bytes)
            st.session_state["single_result"] = {
                "speaker": name, "seed": used_seed, "wav": wav_bytes, "file": fname,
                "model": single_model,
            }

    res = st.session_state.get("single_result")
    if res and res["speaker"] == name:
        st.audio(res["wav"], format="audio/wav")
        st.success(f"生成完了  seed: `{res['seed']}`  モデル: {model_label(res['model'])}  "
                   f"保存先: `{OUTPUTS_DIR / res['file']}`")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("ダウンロード", data=res["wav"], file_name=res["file"],
                               mime="audio/wav", key="single_dl")
        with col2:
            if cfg.get("seed") != res["seed"] and st.button("この声を固定（seedを話者に保存）", key="single_fix"):
                speakers[name]["seed"] = res["seed"]
                # seedは生成したモデルとセットで意味を持つので、モデルも合わせて保存する
                speakers[name]["hf_checkpoint"] = res["model"]
                save_speakers(speakers)
                st.experimental_rerun()
        with col3:
            if st.button("この音声を参照音声として話者に保存", key="single_setref",
                         help="以後この音声を手がかりに生成するので、絵文字を入れても声が安定します。"):
                fname = save_voice_to_library(f"ref_{name}", res["wav"])
                if fname:
                    speakers[name]["ref_wav"] = fname
                    speakers[name]["seed"] = res["seed"]
                    speakers[name]["hf_checkpoint"] = res["model"]
                    save_speakers(speakers)
                    st.experimental_rerun()
    if cfg.get("ref_wav") and st.button("参照音声を解除", key="single_clearref"):
        speakers[name].pop("ref_wav", None)
        save_speakers(speakers)
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
    if "batch_script" not in st.session_state:
        st.session_state["batch_script"] = sample_lines
    emoji_palette("batch_script")
    script_text = st.text_area(
        "台本",
        key="batch_script",
        height=200,
        placeholder="ずんだもん,今日は何の話をしようか？\n四国めたん,天気の話はどうかな。",
    )

    silence_ms = st.slider("発話間の無音（ミリ秒）", 0, 1000, 300, step=50)

    batch_model = select_model_override("batch_model")

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

        def line_model(speaker: str) -> str:
            return speaker_model(speakers[speaker], batch_model)

        # モデルの再読み込みを減らすため、同じモデルの行をまとめて生成する（結合は台本順）
        order = sorted(range(len(lines)), key=lambda i: line_model(lines[i][0]))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            for done, idx in enumerate(order):
                speaker, text = lines[idx]
                hf_repo = line_model(speaker)
                pct = int(done / len(lines) * 100)
                progress.progress(pct, text=f"[{done+1}/{len(lines)}] {speaker}: {text[:20]}...")
                status.info(f"生成中: {speaker} 「{text[:30]}」（{model_label(hf_repo)}）")

                sp_cfg = speakers[speaker]
                ref_wav = sp_cfg.get("ref_wav")
                caption = sp_cfg.get("caption")
                seed = sp_cfg.get("seed")
                no_ref = sp_cfg.get("no_ref", False) or (ref_wav is None and caption is not None)

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

            part_files.sort()
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

    tabv, tab0, tab1, tab2, tab3 = st.tabs(["ボイスデザイン", "単発生成", "話者管理", "バッチ生成", "生成履歴"])
    with tabv:
        page_voicedesign()
    with tab0:
        page_single()
    with tab1:
        page_speakers()
    with tab2:
        page_batch()
    with tab3:
        page_history()


if __name__ == "__main__":
    main()
