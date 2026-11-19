import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
import librosa
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip, vfx, ImageClip
import os
import threading
import queue
import random
import traceback
import time
import tempfile
import subprocess
import shutil
import pygame
import imageio_ffmpeg
import PIL.Image
import PIL.ImageDraw
import json
import zlib
import warnings
warnings.filterwarnings('ignore')

# Drag & Drop ist optional: ohne tkinterdnd2 laeuft alles wie gehabt.
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND_AVAILABLE = True
except Exception:
    TkinterDnD = None
    DND_FILES = None
    _DND_AVAILABLE = False

VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
AUDIO_EXTS = ('.mp3', '.wav', '.ogg', '.aac', '.flac', '.m4a', '.opus')
PROJECT_EXTS = ('.aep',)

# Kein aufblitzendes Konsolenfenster pro ffmpeg-Aufruf (Windows).
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _ffmpeg_tail(stderr_bytes, lines=8):
    """Die letzten Zeilen der ffmpeg-Ausgabe - dort steht der eigentliche Fehler."""
    try:
        text = (stderr_bytes or b"").decode("utf-8", errors="replace")
    except Exception:
        return ""
    rows = [r for r in text.strip().splitlines() if r.strip()]
    return "\n".join(rows[-lines:])


def run_ffmpeg(cmd):
    """ffmpeg ausfuehren; bei Fehler RuntimeError mit lesbarer Ursache."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg ist fehlgeschlagen:\n" + _ffmpeg_tail(res.stderr))
    return res

# =====================================================================
# Compatibility Patches & OpenCV Optimization
# =====================================================================
if not hasattr(PIL.Image, 'ANTIALIAS'):
    try:
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import moviepy.video.io.ffmpeg_reader as ffr
_orig_read_frame = ffr.FFMPEG_VideoReader.read_frame
def _robust_read_frame(self):
    try:
        return _orig_read_frame(self)
    except IOError as e:
        if "failed to read the first frame" in str(e):
            try:
                self.initialize(0.1)
                self.pos = int(self.fps * 0.1 + 0.00001) + 1
                return _orig_read_frame(self)
            except Exception:
                pass
            w, h = self.size
            depth = getattr(self, 'depth', 3)
            blank = np.zeros((h, w, depth), dtype='uint8')
            self.lastread = blank
            return blank
        raise
ffr.FFMPEG_VideoReader.read_frame = _robust_read_frame

# Patch for MoviePy 1.0.3 crop bug: prevent negative coordinate slicing
import moviepy.video.fx.crop as mpy_crop
def _safe_moviepy_crop(clip, x1=None, y1=None, x2=None, y2=None, width=None, height=None, x_center=None, y_center=None):
    cw, ch = clip.size
    if width is not None and x1 is not None: x2 = x1 + width
    elif width is not None and x2 is not None: x1 = x2 - width
    if height is not None and y1 is not None: y2 = y1 + height
    elif height is not None and y2 is not None: y1 = y2 - height

    if x_center is not None and width is not None:
        x1 = x_center - width / 2.0
        x2 = x_center + width / 2.0
    if y_center is not None and height is not None:
        y1 = y_center - height / 2.0
        y2 = y_center + height / 2.0

    if x1 is None: x1 = 0
    if y1 is None: y1 = 0
    if x2 is None: x2 = cw
    if y2 is None: y2 = ch

    tw_req = int(round(x2 - x1)) if width is None else int(round(width))
    th_req = int(round(y2 - y1)) if height is None else int(round(height))
    tw_req = max(1, min(cw, tw_req))
    th_req = max(1, min(ch, th_req))

    ix1 = max(0, min(cw - tw_req, int(round(x1))))
    iy1 = max(0, min(ch - th_req, int(round(y1))))
    ix2 = ix1 + tw_req
    iy2 = iy1 + th_req

    newclip = clip.fl_image(lambda pic: pic[iy1:iy2, ix1:ix2], apply_to=["mask"])
    newclip.size = (tw_req, th_req)
    newclip.w = tw_req
    newclip.h = th_req
    return newclip
mpy_crop.crop = _safe_moviepy_crop

# Ultra-fast SIMD resizer via OpenCV
import moviepy.video.fx.resize as mpy_resize
def _cv2_resizer(pic, newsize):
    lx, ly = max(1, int(round(newsize[0]))), max(1, int(round(newsize[1])))
    return cv2.resize(pic.astype(np.uint8), (lx, ly), interpolation=cv2.INTER_LINEAR)
mpy_resize.resizer = _cv2_resizer

# Pre-generate Cinematic S-Curve Contrast LUT
_x_lut = np.linspace(0, 1, 256)
_y_lut = 1.0 / (1.0 + np.exp(-9.0 * (_x_lut - 0.5)))
_y_lut = (_y_lut - _y_lut.min()) / (_y_lut.max() - _y_lut.min())
CINEMATIC_LUT = np.clip(_y_lut * 255.0, 0, 255).astype(np.uint8)

def _build_vignette_mask(w, h):
    X = cv2.getGaussianKernel(w, w * 0.48)
    Y = cv2.getGaussianKernel(h, h * 0.48)
    kernel = Y * X.T
    kernel = kernel / kernel.max()
    vig = np.clip(0.68 + 0.32 * kernel, 0, 1)
    vig_u8 = np.clip(vig * 255.0, 0, 255).astype(np.uint8)
    return cv2.merge([vig_u8, vig_u8, vig_u8])

def ai_find_drop_candidates(y, sr, total_dur, max_candidates=8, min_gap=1.5):
    """
    KI-Drop-Kandidaten-Analyse fuer einen ausgewaehlten Audio-Bereich.

    Kandidaten-Punkte kommen aus zwei Quellen:
      1. Agglomerative Segmentierung (Chroma-CQT + MFCC) -> musikalische Abschnitte
      2. Steile Energiespruenge im gleitenden Mehrband-Profil

    Bewertet wird jeder Kandidat als Energie-Sprung ueber die Grenze hinweg
    (RMS, Sub-Bass 30-160 Hz, Air 4-11 kHz). Der beste Drop ist die Grenze,
    die in den lautesten nachfolgenden Abschnitt fuehrt - nicht die mit dem
    groessten relativen Sprung. Beginnt der Bereich mit einer stillen Intro,
    gewinnt der erste echte Drop (Intro-Regel).

    Rueckgabe: Liste von Dicts (bestes zuerst) mit time, score, jump,
    before, after, is_best, intro_promoted.
    """
    if total_dur < 3.5:
        return []

    try:
        hop = 1024
        n_fft = 2048

        spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        band_sub = spec[(freqs >= 30) & (freqs < 160)].mean(axis=0)
        band_air = spec[(freqs >= 4000) & (freqs < 11000)].mean(axis=0)
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]

        n_frames = len(rms)
        frames_t = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)

        smooth_k = max(1, int(sr / hop * 0.8) | 1)

        def _band(arr):
            sm = np.convolve(arr, np.ones(smooth_k) / smooth_k, mode="same")
            return sm / (float(np.percentile(sm, 95)) + 1e-9)

        p_rms, p_sub, p_air = _band(rms), _band(band_sub), _band(band_air)

        raw_candidates = set()
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
            n_clusters = max(3, min(8, int(total_dur // 7)))
            bounds = librosa.segment.agglomerative(np.vstack([chroma, mfcc]), n_clusters)
            raw_candidates.update(
                round(float(b), 2)
                for b in librosa.frames_to_time(bounds, sr=sr, hop_length=hop)
            )
        except Exception:
            pass

        for i in range(4, n_frames - 8):
            a_r = float(np.mean(p_rms[i - 4:i + 5]))
            b_r = float(np.mean(p_rms[i:i + 9]))
            a_s = float(np.mean(p_sub[i - 4:i + 5]))
            b_s = float(np.mean(p_sub[i:i + 9]))
            if b_r > a_r * 1.15 and b_s > a_s * 1.05:
                raw_candidates.add(round(float(frames_t[i]), 2))

        raw_candidates = sorted(c for c in raw_candidates if 1.2 < c < total_dur * 0.90)
        if not raw_candidates:
            return []

        win = max(6, int(sr / hop * 1.2))

        def _window_mean(arr, i0, i1):
            i0, i1 = max(0, i0), min(n_frames, i1)
            return float(np.mean(arr[i0:i1])) if i1 > i0 else 0.0

        def _measure(t_cand):
            i = int(np.searchsorted(frames_t, t_cand))
            pre_r, post_r = _window_mean(p_rms, i - win, i), _window_mean(p_rms, i, i + win)
            pre_s, post_s = _window_mean(p_sub, i - win, i), _window_mean(p_sub, i, i + win)
            pre_a, post_a = _window_mean(p_air, i - win, i), _window_mean(p_air, i, i + win)
            jump = (max(0.0, post_r - pre_r)
                    + 1.2 * max(0.0, post_s - pre_s)
                    + 0.5 * max(0.0, post_a - pre_a))
            after = post_r + 0.45 * post_s + 0.15 * post_a
            before = pre_r + 0.45 * pre_s + 0.15 * pre_a
            return jump, after, before

        scored = []
        for c in raw_candidates:
            jump, after, before = _measure(c)
            if jump < 0.30:
                continue
            if after <= before * 1.03 or after < 0.35:
                continue
            scored.append({
                "time": c,
                "jump": jump,
                "after": after,
                "before": before,
                "level": after - 0.35 * before,
            })

        if not scored:
            return []

        scored.sort(key=lambda d: d["time"])

        def _merge(items, gap, key):
            """Cluster zusammenfassen; Cluster-Vertreter ist der mit dem
            hoechsten key. Die Reihenfolge bleibt zeitlich sortiert."""
            out = []
            for item in sorted(items, key=lambda d: d["time"]):
                if out and (item["time"] - out[-1]["time"]) < gap:
                    if key(item) > key(out[-1]):
                        out[-1] = item
                    continue
                out.append(item)
            return out

        # Strikte Definition = echte Drop-Kandidaten (bestimmt den Gewinner)
        strict = [d for d in scored
                  if d["jump"] >= 0.50 and d["after"] > d["before"] * 1.05 and d["after"] >= 0.45]
        strict_merged = _merge(strict, min_gap, lambda d: d["level"])

        # Erweiterte Liste = auch schwächere Uebergaenge, fuer Alternativen
        extended = _merge(scored, 1.2, lambda d: d["jump"])

        pool = list(strict_merged)
        for item in extended:
            if any(abs(item["time"] - existing["time"]) < 1.0 for existing in pool):
                continue
            pool.append(item)
        pool.sort(key=lambda d: d["time"])

        if not pool:
            return []

        # Intro-Regel: beginnt der Bereich still, gewinnt der ERSTE Kandidat.
        # Wird bewusst aus der ungemergten strikten Liste genommen (nicht aus
        # strict_merged) - sonst verschiebt das Cluster-Merge den Gewinner um
        # mehrere Hundertstel Sekunden nach hinten.
        intro_promoted = False
        best_item = None
        if strict:
            first_strict = strict[0]
            fi = int(np.searchsorted(frames_t, first_strict["time"]))
            if _window_mean(p_rms, max(0, fi - 2 * win), max(1, fi)) < 0.35:
                intro_promoted = True
                best_item = next(
                    (d for d in pool if abs(d["time"] - first_strict["time"]) < 0.4),
                    first_strict
                )
                if best_item not in pool:
                    pool.append(best_item)

        for item in pool:
            item["raw_time"] = item["time"]
            item["time"] = snap_candidate_to_transient(
                item["time"], p_sub, p_rms, frames_t
            )

        pool.sort(key=lambda d: d["time"])

        if best_item is None:
            best_idx = int(np.argmax([d["level"] for d in pool]))
        else:
            best_idx = next((i for i, d in enumerate(pool) if d is best_item), 0)

        # Besten zuerst: Intro-Regel und Level-Ranking duerfen nicht
        # auseinanderlaufen, sonst zeigt die UI einen anderen als den benutzten.
        best = pool[best_idx]
        rest = [d for i, d in enumerate(pool) if i != best_idx]
        rest.sort(key=lambda d: -d["level"])
        ranked = [best] + rest

        top_level = ranked[0]["level"] or 1.0
        for idx, item in enumerate(ranked):
            item["is_best"] = (idx == 0)
            item["intro_promoted"] = bool(intro_promoted and idx == 0)
            item["confidence"] = float(max(0.0, min(1.0, item["level"] / top_level)))

        return ranked[:max_candidates]
    except Exception as e:
        print(f"[AI Drop Candidate Error] {e}")
        return []


def snap_candidate_to_transient(t_cand, spec_p, rms_p, frames_t, radius_frames=6):
    """Zieht einen Kandidaten auf den staerksten Attack im Fenster."""
    try:
        idx = int(np.searchsorted(frames_t, t_cand))
        lo = max(0, idx - radius_frames)
        hi = min(len(frames_t), idx + radius_frames + 1)
        if hi <= lo:
            return float(t_cand)
        strength = 0.6 * spec_p[lo:hi] + 0.4 * rms_p[lo:hi]
        return float(frames_t[lo + int(np.argmax(strength))])
    except Exception:
        return float(t_cand)


def ai_detect_beat_drop(y, sr, total_dur):
    """
    KI-basierte Struktur-Analyse & Beat-Drop Erkennung (Change-Point Detection).
    Liefert nur den besten Kandidaten; see ai_find_drop_candidates fuer Details.
    """
    found = ai_find_drop_candidates(y, sr, total_dur, max_candidates=1)
    return float(found[0]["time"]) if found else 0.0


def compute_waveform_envelope(audio_path, offset=0.0, duration=None, sr=22050, block_s=0.02):
    """Absolut-Peak-Huelle fuer die Wellenform (ohne Beat-/Drop-Analyse)."""
    y, _ = librosa.load(audio_path, sr=sr, offset=offset, duration=duration, mono=True)
    if len(y) < sr // 4:
        return None, block_s
    block = max(64, int(sr * block_s))
    n_blocks = len(y) // block
    if n_blocks < 2:
        return None, block_s
    env = np.abs(y[:n_blocks * block].reshape(n_blocks, block)).max(axis=1)
    return env, block / float(sr)


def snap_beats_to_transients(beat_times, onset_env, sr, hop=512, max_shift=0.06):
    """
    Jitter-Korrektur: Verschiebt jeden Beat auf den staerksten Transient
    in einem Fenster von +-max_shift Sekunden. Entfernt Timing-Flackern,
    die durch ungenauen Beat-Grid und Keyframe-Quantisierung entstehen.
    """
    if onset_env is None or len(onset_env) == 0 or len(beat_times) == 0:
        return [float(b) for b in beat_times]

    hop_sec = hop / float(sr)
    radius = max(1, int(round(max_shift / hop_sec)))
    n_env = len(onset_env)
    snapped = []

    for b in beat_times:
        b = float(b)
        center = int(round(b / hop_sec))
        lo = max(0, center - radius)
        hi = min(n_env, center + radius + 1)
        if hi <= lo:
            snapped.append(b)
            continue
        peak_idx = lo + int(np.argmax(onset_env[lo:hi]))
        peak_t = peak_idx * hop_sec
        if abs(peak_t - b) <= max_shift:
            snapped.append(peak_t)
        else:
            snapped.append(b)

    snapped.sort()
    return snapped


def ai_find_best_moment(audio_path, target_len, sr=22050, max_scan=900.0):
    """
    Best-Moment-Finder: Sucht die energiereichste Passage des Songs und
    bevorzugt Fenster, in denen der Beat-Drop in den ersten Sekunden liegt.
    Liefert (start, end, drop) oder None.
    """
    try:
        dur_total = float(librosa.get_duration(path=audio_path))
        if dur_total < 3.0:
            return None

        offset = max(0.0, dur_total - max_scan)
        y, _ = librosa.load(audio_path, sr=sr, offset=offset, duration=max_scan)
        if len(y) < sr:
            return None
        scan_dur = len(y) / float(sr)

        hop = 1024
        n_fft = 2048
        spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        sub = spec[(freqs >= 30) & (freqs < 160)].mean(axis=0)
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]

        ft = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

        def _smooth(a, sec=1.0):
            k = max(1, int(sr / hop * sec) | 1)
            return np.convolve(a, np.ones(k) / k, mode="same")

        e = 0.65 * (_smooth(rms) / (np.percentile(rms, 95) + 1e-9)) + \
            0.35 * (_smooth(sub) / (np.percentile(sub, 95) + 1e-9))

        drop_rel = ai_detect_beat_drop(y, sr, scan_dur)
        if drop_rel <= 0.05:
            drop_rel = None

        step = 0.5
        win_frames = max(2, int(target_len / (hop / float(sr))))
        best = None
        for start_i in range(0, max(1, len(e) - win_frames), int(step / (hop / float(sr)))):
            seg = e[start_i:start_i + win_frames]
            if len(seg) < win_frames * 0.5:
                break
            t_rel = float(ft[start_i])
            score = 1.40 * float(np.mean(seg))
            score += 0.50 * float(np.percentile(seg, 85))
            head = e[start_i:start_i + int(3.0 / (hop / float(sr)))]
            if len(head):
                score += 0.45 * float(np.max(head))
            if drop_rel is not None:
                if t_rel <= drop_rel < (t_rel + target_len):
                    offset_in = drop_rel - t_rel
                    score += 1.10 - (min(offset_in, 12.0) / 12.0) * 0.55
            if best is None or score > best[0]:
                best = (score, t_rel)

        if best is None:
            return None

        start_rel = max(0.0, min(scan_dur - target_len, best[1]))
        start_abs = start_rel + offset
        end_abs = min(dur_total, start_abs + target_len)
        drop_abs = (drop_rel + offset) if drop_rel is not None else 0.0
        return (float(start_abs), float(end_abs), float(drop_abs))
    except Exception as e:
        print(f"[Best Moment Error] {e}")
        return None


class FaceReframer:
    """
    KI-Auto-Reframe: haelt das Subject im 9:16-Crop.

    Backends (in dieser Reihenfolge):
      1. OpenCV YuNet DNN-Gesichtserkennung, falls ein ONNX-Modell im
         Programmordner liegt (face_detection_yunet_*.onnx)
      2. Hautton-Segmentierung (YCrCb) groesster Person-Blob
      3. Bewegungszentrum (Frame-Differenz) als letzte Reserve

    Das Ergebnis wird zeitlich geglaettet, damit das Bild nicht zappelt.
    """
    DETECT_W = 320
    MODEL_PATTERNS = ("face_detection_yunet_*.onnx", "yunet_*.onnx")

    def __init__(self):
        self.yn = None
        self._prev_gray = None
        self._smooth_x = None
        self.backend = "none"

        for pattern in self.MODEL_PATTERNS:
            try:
                import glob
                base = os.path.dirname(os.path.abspath(__file__))
                matches = glob.glob(os.path.join(base, pattern))
                if matches and hasattr(cv2, "FaceDetectorYN"):
                    self.yn = cv2.FaceDetectorYN.create(matches[0], "", (320, 320), 0.6, 0.3, 5000)
                    self.backend = "yunet"
                    break
            except Exception as e:
                print(f"[Auto-Reframe] YuNet konnte nicht geladen werden: {e}")
                self.yn = None

        if self.yn is None:
            self.backend = "skin"
        self._skin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _detect_yunet(self, small_bgr):
        try:
            h, w = small_bgr.shape[:2]
            self.yn.setInputSize((w, h))
            _, faces = self.yn.detect(small_bgr)
            if faces is None or len(faces) == 0:
                return None
            best = max(faces, key=lambda f: f[2] * f[3])
            return float(np.clip((best[0] + best[2] / 2.0) / w, 0.0, 1.0))
        except Exception:
            return None

    def _detect_skin(self, small_bgr, orig_w):
        try:
            ycc = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycc, (0, 133, 77), (255, 182, 128))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._skin_kernel)
            mask = cv2.dilate(mask, self._skin_kernel, iterations=2)
            h, w = mask.shape[:2]
            min_area = max(40, int(w * h * 0.004))
            count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            best_area = 0
            best_x = None
            for i in range(1, count):
                x, y, bw, bh, area = stats[i]
                if area < min_area:
                    continue
                if bh < h * 0.12:
                    continue
                fill = area / float(max(1, bw * bh))
                if fill < 0.32:
                    continue
                if area > best_area:
                    best_area = area
                    best_x = centroids[i][0]
            if best_x is None:
                return None
            return float(np.clip(best_x / w, 0.0, 1.0))
        except Exception:
            return None

    def _detect_motion(self, gray):
        try:
            if self._prev_gray is None or self._prev_gray.shape != gray.shape:
                self._prev_gray = gray
                return None
            diff = cv2.absdiff(gray, self._prev_gray)
            self._prev_gray = gray
            if float(diff.mean()) < 2.0:
                return None
            col = diff.astype(np.float32).mean(axis=0)
            total = float(col.sum())
            if total < 1.0:
                return None
            xs = np.arange(len(col), dtype=np.float32)
            return float(np.clip((xs * col).sum() / total / max(1, len(col) - 1), 0.0, 1.0))
        except Exception:
            return None

    def center_ratio(self, frame_bgr):
        try:
            h, w = frame_bgr.shape[:2]
            scale = self.DETECT_W / float(w)
            sw = self.DETECT_W
            sh = max(1, int(h * scale))
            small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)

            if self.yn is not None:
                ratio = self._detect_yunet(small)
                if ratio is not None:
                    return ratio

            ratio = self._detect_skin(small, w)
            if ratio is not None:
                return ratio

            return self._detect_motion(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        except Exception:
            return None

    def track_center(self, frame_bgr, alpha=0.28):
        raw = self.center_ratio(frame_bgr)
        if raw is None:
            return self._smooth_x
        if self._smooth_x is None:
            self._smooth_x = raw
        else:
            self._smooth_x = (1.0 - alpha) * self._smooth_x + alpha * raw
        return self._smooth_x


def compute_audio_overview(audio_path, sr=22050, max_dur=1200.0):
    """
    Liefert (env, hop_sec, beat_times, drop_time) fuer die Wellenform-Ansicht.
    env: Absolut-Peak-Huelle, beat_times: Taktgitter, drop_time: KI-Drop.
    """
    y, _ = librosa.load(audio_path, sr=sr, duration=max_dur, mono=True)
    if len(y) < sr // 2:
        return None, 0.0, [], 0.0

    hop = 512
    block = max(256, int(sr * 0.02))
    n_blocks = len(y) // block
    if n_blocks < 4:
        return None, 0.0, [], 0.0
    env = np.abs(y[:n_blocks * block].reshape(n_blocks, block)).max(axis=1)

    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
        beat_times = list(librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop))
    except Exception:
        beat_times = []

    try:
        drop_time = ai_detect_beat_drop(y, sr, len(y) / float(sr))
    except Exception:
        drop_time = 0.0

    return env, block / float(sr), beat_times, float(drop_time)


def format_eta(seconds):
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


# =====================================================================
# Effekt-Engine: Bausteine & Stil-Profile
# =====================================================================

EFFECT_SETS = {
    "CapCut Kinetic (Empfohlen)": {
        "punch": 0.045, "punch_dur": 0.16,
        "whip_prob": 0.10, "whip_px": 0.055,
        "flash_on_accent": True, "flash_max": 0.22,
        "shake_on_accent": True, "shake_px": 3.0,
        "zoom_blur_on_drop": True,
        "freeze_on_drop": True,
        "grain": 0.010,
        "vignette": True,
        "collage_prob": 0.05,
        "spin_prob": 0.010,
    },
    "Cinematic (Zurückhaltend)": {
        "punch": 0.028, "punch_dur": 0.14,
        "whip_prob": 0.05, "whip_px": 0.035,
        "flash_on_accent": True, "flash_max": 0.14,
        "shake_on_accent": False, "shake_px": 0.0,
        "zoom_blur_on_drop": True,
        "freeze_on_drop": False,
        "grain": 0.014,
        "vignette": True,
        "collage_prob": 0.02,
        "spin_prob": 0.0,
    },
    "Beat Edit (Dynamisch)": {
        "punch": 0.070, "punch_dur": 0.18,
        "whip_prob": 0.22, "whip_px": 0.075,
        "flash_on_accent": True, "flash_max": 0.34,
        "shake_on_accent": True, "shake_px": 5.0,
        "zoom_blur_on_drop": True,
        "freeze_on_drop": True,
        "grain": 0.008,
        "vignette": True,
        "collage_prob": 0.09,
        "spin_prob": 0.018,
    },
    "Maximal (Volle Power)": {
        "punch": 0.095, "punch_dur": 0.20,
        "whip_prob": 0.34, "whip_px": 0.095,
        "flash_on_accent": True, "flash_max": 0.48,
        "shake_on_accent": True, "shake_px": 8.0,
        "zoom_blur_on_drop": True,
        "freeze_on_drop": True,
        "grain": 0.006,
        "vignette": True,
        "collage_prob": 0.14,
        "spin_prob": 0.028,
    },
    "Keine Effekte (Clean Cut)": {
        "punch": 0.0, "punch_dur": 0.0,
        "whip_prob": 0.0, "whip_px": 0.0,
        "flash_on_accent": False, "flash_max": 0.0,
        "shake_on_accent": False, "shake_px": 0.0,
        "zoom_blur_on_drop": False,
        "freeze_on_drop": False,
        "grain": 0.006,
        "vignette": True,
        "collage_prob": 0.0,
        "spin_prob": 0.0,
    },
}


def get_effect_set(name):
    base = EFFECT_SETS.get(name)
    if base is None:
        base = EFFECT_SETS["CapCut Kinetic (Empfohlen)"]
    return dict(base)


def build_motion_blur_kernel(length, angle_deg):
    """Lineare Bewegungsunschärfe als Kernel fuer cv2.filter2D."""
    length = max(3, int(length) | 1)
    if length % 2 == 0:
        length += 1
    k = np.zeros((length, length), np.float32)
    c = length // 2
    rad = np.deg2rad(float(angle_deg))
    dx, dy = np.cos(rad), np.sin(rad)
    for i in range(length):
        step = (i - c) / float(max(1, c))
        x = int(round(c + dx * step * c))
        y = int(round(c + dy * step * c))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] += 1.0
    total = float(k.sum())
    if total <= 0:
        return np.full((length, length), 1.0 / (length * length), np.float32)
    return k / total


def motion_blur_frame(frame, length, angle_deg):
    if length < 3:
        return frame
    kernel = build_motion_blur_kernel(length, angle_deg)
    return cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_REFLECT)


def radial_zoom_blur(frame, amount):
    """Radialer Zoom-Blur - sieht nach Geschwindigkeit aus, nicht nach Filter."""
    if amount <= 0.002:
        return frame
    h, w = frame.shape[:2]
    s = 1.0 + amount
    nh, nw = max(h, int(round(h * s))), max(w, int(round(w * s)))
    big = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = (nw - w) // 2
    y0 = (nh - h) // 2
    zoomed = big[y0:y0 + h, x0:x0 + w]
    if zoomed.shape[0] != h or zoomed.shape[1] != w:
        return frame
    a = min(0.8, amount * 1.25)
    return cv2.addWeighted(frame, 1.0 - a, zoomed, a, 0.0)


def build_grain_field(h, w, amount, seed):
    """
    Statisches Filmkorn-Feld.

    Wichtig: das Korn muss ueber die Frames GLEICH bleiben. Pro Frame neu
    gewuerfelt zerstoert die Interframe-Kompression voellig - dieselbe
    Renderzeit kostet ~8x Dateigroesse.
    """
    if amount <= 0.0:
        return None
    rng = np.random.default_rng(seed)
    tile = rng.normal(0.0, 255.0 * amount, (256, 256)).astype(np.float32)
    reps_y = int(np.ceil(h / 256)) + 1
    reps_x = int(np.ceil(w / 256)) + 1
    field = np.tile(tile, (reps_y, reps_x))
    return np.ascontiguousarray(field[:h + 256, :w + 256])


def apply_film_grain(frame, field, frame_index=0):
    """
    Statisches Korn. Bewusst kein Rollen pro Frame: das wuerde die
    Interframe-Kompression zerstoeren und die Datei um ein Vielfaches
    aufblaehen. Optisch ist der Unterschied bei 1% kaum sichtbar.
    """
    if field is None:
        return frame
    h, w = frame.shape[:2]
    noise = field[:h, :w]
    if noise.shape[0] != h or noise.shape[1] != w:
        noise = np.resize(noise, (h, w))
    out = frame.astype(np.float32)
    out += noise[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def build_effect_plan(cut_idx, total_cuts, cut_global_start, kick_strength,
                     is_drop_cut, cfg, intensity, rng):
    """
    Legt fest, was EIN Cut bekommt - bewusst sparsam.

    Grundregel: die Mehrheit der Cuts bleibt sauber. Effekte landen auf
    musikalischen Positionen (Drop, Akzente) und nicht auf jedem Schnitt,
    sonst wirkt das Ergebnis wie eine PowerPoint-Folienfolge.
    """
    punch = cfg["punch"] * (0.55 + 0.45 * intensity)
    whip_px = cfg["whip_px"] * (0.55 + 0.45 * intensity)
    shake_px = cfg["shake_px"] * (0.55 + 0.45 * intensity)
    punch_dur = cfg["punch_dur"]

    plan = {
        "transform": "none",
        "amount": 0.0,
        "duration": 0.0,
        "angle": 0.0,
        "flash": 0.0,
        "shake": 0.0,
        "zoom_blur": 0.0,
        "blur_len": 0,
    }

    # Impact-Rhythmus: Akzent-Effekte nur auf jedem n-ten Cut
    cadence = max(1, int(cfg.get("cadence", 1)))
    accent_gate = (cut_idx % cadence) == 0

    # Whip-Pan als "Uebergang" - Richtungen wechseln, damit es nicht als Muster lesbar wird
    if whip_px > 0.0 and accent_gate and rng.random() < cfg["whip_prob"]:
        plan["transform"] = "whip_pan"
        plan["amount"] = whip_px
        plan["duration"] = min(0.16, punch_dur + 0.04)
        plan["angle"] = 0.0 if (cut_idx % 2 == 0) else 180.0
        plan["blur_len"] = 9
        return plan

    # Drop: Zoom-Blur-Burst statt Dauerbewegung
    if is_drop_cut and cfg["zoom_blur_on_drop"]:
        plan["transform"] = "punch_in"
        plan["amount"] = max(punch * 1.4, 0.03)
        plan["duration"] = max(0.12, punch_dur * 0.8)
        plan["zoom_blur"] = 0.10
        plan["flash"] = min(0.5, max(0.10, kick_strength * 0.45))
        return plan

    # Akzent-Schnitte: Punch-In + optional Flash
    accent = accent_gate and kick_strength >= 0.45
    if accent and punch > 0.0:
        plan["transform"] = "punch_in" if (cut_idx // 4) % 2 == 0 else "punch_out"
        plan["amount"] = punch * (0.75 + 0.35 * kick_strength)
        plan["duration"] = min(punch_dur, 0.45 * punch_dur + 0.4 * kick_strength * punch_dur)

    if cfg["flash_on_accent"] and accent and kick_strength >= 0.5:
        plan["flash"] = min(cfg["flash_max"], kick_strength * cfg["flash_max"])

    if cfg["shake_on_accent"] and accent and shake_px > 0.0 and kick_strength >= 0.75:
        plan["shake"] = shake_px * kick_strength

    return plan


def build_metronome_click(sr=44100):
    """Kurzer, perkussiver Klick fuer die Latenz-Kalibrierung."""
    dur = 0.055
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    body = 0.55 * np.sin(2 * np.pi * 1650.0 * t) * np.exp(-t * 90.0)
    tick = 0.45 * np.sin(2 * np.pi * 5200.0 * t) * np.exp(-t * 260.0)
    mono = np.clip(body + tick, -1.0, 1.0)
    stereo = np.stack([mono, mono], axis=1)
    data = (stereo * 32767 * 0.85).astype(np.int16)
    try:
        return pygame.sndarray.make_sound(np.ascontiguousarray(data))
    except Exception as e:
        print(f"[Calibration] Klick-Sound konnte nicht erzeugt werden: {e}")
        return None

# =====================================================================
# Auswahl-Optionen (eine Quelle fuer UI, Presets und Normalisierung)
# =====================================================================
PACE_OPTIONS = ["1x (Jeder Takt)", "2x Schnell (Halber Takt)", "0.5x Entspannt (Doppel-Takt)"]
QUALITY_OPTIONS = ["High Quality (CRF 16 Studio)", "Maximal (CRF 14 Glasklar)", "Ultra-Fast (CRF 20 Schnell)"]
RES_OPTIONS = ["Auto (Quell-Auflösung)", "1080p (Full HD)", "1440p (2K Ultra Sharp)", "Originalgröße der Dateien"]
CLIP_AUDIO_OPTIONS = ["Stumm (100% Musik)", "Dezent 15% (Killsounds/SFX)", "Mittel 35% (Dialoge/SFX)"]
PREROLL_OPTIONS = [
    "Aus (Edit startet am Drop)",
    "2 Beats davor",
    "4 Beats davor (1 Takt)",
    "8 Beats davor (2 Takte)",
    "16 Beats davor (4 Takte)",
    "32 Beats davor (8 Takte)",
]
CADENCE_OPTIONS = ["Jeder Cut (Maximal)", "Jeder 2. Cut (Sauberer Takt)", "Jeder 4. Cut (Nur Akzente)"]
OVERLAY_CORNER_OPTIONS = [
    "Wechselt zum Takt (Animiert)", "In der Mitte (Zentriert)",
    "Oben Links", "Oben Rechts", "Unten Links", "Unten Rechts",
]
OVERLAY_SIZE_OPTIONS = ["Kompakt (20% - Dezent)", "Mittel (30% - Standard)", "Groß (45% - Fett & Dominant)"]
BEST_MOMENT_OPTIONS = ["15s", "30s", "60s", "90s"]


def _match_option(value, options, keywords=None, default=None):
    """
    Bringt einen gespeicherten Wert auf eine aktuell gueltige Option.

    Aeltere presets.json/.aep-Dateien enthalten Texte, die es im UI nicht
    mehr gibt ("Gross (45% Bildhoehe)", kaputte Umlaute ...). Ohne diese
    Abbildung zeigte das Dropdown dann einen Wert, den keine Render-Logik
    kannte.
    """
    if value in options:
        return value
    text = str(value or "").lower()
    for kw, opt in (keywords or []):
        if kw in text:
            return opt
    return default if default is not None else options[0]


def normalize_preset(data):
    """Validiert ein Preset-Dict; unbekannte Werte werden sinnvoll ersetzt."""
    if not isinstance(data, dict):
        return {}
    d = dict(data)
    if "pace" in d:
        d["pace"] = _match_option(d["pace"], PACE_OPTIONS,
                                  [("2x", PACE_OPTIONS[1]), ("0.5", PACE_OPTIONS[2])])
    if "quality" in d:
        d["quality"] = _match_option(d["quality"], QUALITY_OPTIONS,
                                     [("crf 14", QUALITY_OPTIONS[1]), ("maximal", QUALITY_OPTIONS[1]),
                                      ("ultra", QUALITY_OPTIONS[2]), ("crf 20", QUALITY_OPTIONS[2])])
    if "resolution" in d:
        d["resolution"] = _match_option(d["resolution"], RES_OPTIONS,
                                        [("auto", RES_OPTIONS[0]), ("1440", RES_OPTIONS[2]),
                                         ("1080", RES_OPTIONS[1]), ("original", RES_OPTIONS[3])])
    if "clip_audio" in d:
        d["clip_audio"] = _match_option(d["clip_audio"], CLIP_AUDIO_OPTIONS,
                                        [("15", CLIP_AUDIO_OPTIONS[1]), ("35", CLIP_AUDIO_OPTIONS[2])])
    if "preroll" in d:
        d["preroll"] = _match_option(d["preroll"], PREROLL_OPTIONS,
                                     [("32", PREROLL_OPTIONS[5]), ("16", PREROLL_OPTIONS[4]),
                                      ("8 ", PREROLL_OPTIONS[3]), ("4 ", PREROLL_OPTIONS[2]),
                                      ("2 ", PREROLL_OPTIONS[1]), ("aus", PREROLL_OPTIONS[0])])
    if "fx_cadence" in d:
        d["fx_cadence"] = _match_option(d["fx_cadence"], CADENCE_OPTIONS,
                                        [("4.", CADENCE_OPTIONS[2]), ("2.", CADENCE_OPTIONS[1]),
                                         ("jeder cut", CADENCE_OPTIONS[0])],
                                        default=CADENCE_OPTIONS[1])
    if "effect_set" in d:
        d["effect_set"] = _match_option(d["effect_set"], list(EFFECT_SETS.keys()),
                                        [("keine", "Keine Effekte (Clean Cut)"),
                                         ("cinematic", "Cinematic (Zurückhaltend)"),
                                         ("maximal", "Maximal (Volle Power)"),
                                         ("beat", "Beat Edit (Dynamisch)")],
                                        default="CapCut Kinetic (Empfohlen)")
    if "overlay_corner" in d:
        d["overlay_corner"] = _match_option(d["overlay_corner"], OVERLAY_CORNER_OPTIONS,
                                            [("wechsel", OVERLAY_CORNER_OPTIONS[0]),
                                             ("mitte", OVERLAY_CORNER_OPTIONS[1])])
    if "overlay_size" in d:
        d["overlay_size"] = _match_option(d["overlay_size"], OVERLAY_SIZE_OPTIONS,
                                          [("kompakt", OVERLAY_SIZE_OPTIONS[0]), ("20%", OVERLAY_SIZE_OPTIONS[0]),
                                           ("mittel", OVERLAY_SIZE_OPTIONS[1]), ("30%", OVERLAY_SIZE_OPTIONS[1]),
                                           ("gro", OVERLAY_SIZE_OPTIONS[2]), ("45%", OVERLAY_SIZE_OPTIONS[2])],
                                          default=OVERLAY_SIZE_OPTIONS[1])
    if d.get("overlay_start_mode") not in (None, "start", "drop", "custom"):
        d["overlay_start_mode"] = "drop"
    if d.get("mode") not in (None, "auto", "onsets", "manual"):
        d["mode"] = "auto"
    if d.get("orientation") not in (None, "original", "portrait", "landscape"):
        d["orientation"] = "portrait"
    if d.get("scale_mode") not in (None, "fit", "fill"):
        d["scale_mode"] = "fill"
    if "fps" in d:
        try:
            d["fps"] = 30 if int(d["fps"]) <= 30 else 60
        except Exception:
            d["fps"] = 60
    return d


def _preset(**kw):
    base = {
        "mode": "auto",
        "pace": PACE_OPTIONS[0],
        "orientation": "portrait",
        "scale_mode": "fill",
        "fps": 60,
        "quality": QUALITY_OPTIONS[0],
        # Keine Hochskalierung per Default - siehe README.
        "resolution": RES_OPTIONS[0],
        "clip_audio": CLIP_AUDIO_OPTIONS[0],
        "fade_out": True,
        "auto_loudness": True,
        "snap_jitter": True,
        "ai_reframe": False,
        "preroll": PREROLL_OPTIONS[3],
        "effect_set": "CapCut Kinetic (Empfohlen)",
        "fx_cadence": CADENCE_OPTIONS[1],
        "fx_intensity": 100,
        "fx_shake": True,
        "fx_bloom_flash": True,
        "fx_punch_zoom": True,
        "fx_speedramp": True,
        "fx_zoom_blur": True,
        "fx_cinematic_lut": True,
        "fx_vignette": True,
        "fx_grain": True,
        "fx_collage": True,
        "fx_spin": True,
        "enable_overlay": False,
        "overlay_corner": OVERLAY_CORNER_OPTIONS[1],
        "overlay_size": OVERLAY_SIZE_OPTIONS[1],
        "overlay_speed": 1.2,
        "overlay_opacity": 100,
        "overlay_start_mode": "drop",
        "overlay_circle": True,
    }
    base.update(kw)
    return base


DEFAULT_PRESETS = {
    "TikTok Montage / Beat Drop": _preset(
        fx_intensity=130,
        overlay_size=OVERLAY_SIZE_OPTIONS[2],
        overlay_speed=1.8,
    ),
    "Velocity / Action Edit": _preset(
        mode="onsets",
        pace=PACE_OPTIONS[1],
        clip_audio=CLIP_AUDIO_OPTIONS[1],
        preroll=PREROLL_OPTIONS[2],
        effect_set="Beat Edit (Dynamisch)",
        fx_cadence=CADENCE_OPTIONS[0],
        overlay_corner=OVERLAY_CORNER_OPTIONS[0],
    ),
    "Cinematic Flow & Aesthetic": _preset(
        orientation="landscape",
        quality=QUALITY_OPTIONS[1],
        preroll=PREROLL_OPTIONS[4],
        effect_set="Cinematic (Zurückhaltend)",
        fx_intensity=70,
        fx_shake=False,
        fx_punch_zoom=False,
        fx_speedramp=False,
        fx_spin=False,
        overlay_corner="Oben Rechts",
        overlay_size=OVERLAY_SIZE_OPTIONS[0],
        overlay_speed=1.0,
        overlay_opacity=85,
        overlay_start_mode="start",
    ),
    "Clean & Minimal (Pure Cuts)": _preset(
        preroll=PREROLL_OPTIONS[0],
        effect_set="Keine Effekte (Clean Cut)",
        fx_cadence=CADENCE_OPTIONS[2],
        fx_intensity=40,
        fx_shake=False,
        fx_bloom_flash=False,
        fx_punch_zoom=False,
        fx_speedramp=False,
        fx_zoom_blur=False,
        fx_cinematic_lut=False,
        fx_vignette=False,
        fx_grain=False,
        fx_collage=False,
        fx_spin=False,
        overlay_start_mode="start",
    ),
}

# =====================================================================
# Theme & Configuration
# =====================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


_APP_BASES = (ctk.CTk, TkinterDnD.DnDWrapper) if _DND_AVAILABLE else (ctk.CTk,)


class EditMakerApp(*_APP_BASES):
    def __init__(self):
        super().__init__()

        self.title("Auto Beat-Sync Edit Maker (Pro Studio Edition)")

        self.dnd_enabled = False
        if _DND_AVAILABLE:
            try:
                self.TkdndVersion = TkinterDnD._require(self)
                self.dnd_enabled = True
            except Exception as e:
                print(f"[Drag & Drop] nicht verfuegbar: {e}")

        self.video_paths = []
        self.audio_path = None
        self.audio_duration = 0.0
        self.is_preview_playing = False
        self.manual_times = []
        # Wiederverwendbare Cut-Sammlung: ueberlebt bis zum Rendern, wird mit dem
        # Projekt gespeichert und steuert Render, Vorschau und Batch-Export.
        self.manual_cuts = []
        self.manual_cuts_audio = None
        self.manual_cuts_range = None
        self.manual_cuts_info = {}
        self.manual_review_source = None
        self.is_cancelled = False
        self.last_rendered_path = None
        self.overlay_img_path = None
        self.source_image_cache = {}
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self.current_preview_file = None

        self.undo_stack = []
        self.redo_stack = []
        self._suspend_history = False
        self._history_job = None

        self.render_finished = threading.Event()
        self.render_finished.set()

        self.wave_analysis_running = False
        self._done_win = None

        self.presets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json")
        self.all_presets = self._load_presets_from_disk()
        self.app_settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_settings.json")
        self.app_settings = self._load_app_settings()

        self.batch_running = False
        self._media_info_cache = {}

        # Ohne Audiogeraet (z. B. Kopfhoerer abgezogen, RDP) darf die App
        # trotzdem starten - Rendern braucht pygame nicht.
        self.audio_ok = True
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(16)
        except Exception as e:
            self.audio_ok = False
            print(f"[Audio] pygame.mixer nicht verfuegbar: {e}")

        self._ui_queue = queue.Queue()
        self._build_ui()
        self._last_state = self.capture_full_state()
        self._register_history_traces()
        self._bind_shortcuts()
        self._setup_drag_and_drop()
        self.after(40, self._drain_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_app_close)

    def _load_presets_from_disk(self):
        presets = {k: dict(v) for k, v in DEFAULT_PRESETS.items()}
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for name, values in data.items():
                        if not isinstance(values, dict):
                            continue
                        # Fehlende Keys aelterer Dateien mit den Standardwerten
                        # auffuellen, sonst bleibt z. B. das Effekt-Set eines
                        # vorher gewaehlten Profils einfach stehen.
                        merged = dict(DEFAULT_PRESETS.get(name) or _preset())
                        merged.update(values)
                        presets[name] = normalize_preset(merged)
            except Exception as e:
                print(f"Fehler beim Laden von presets.json: {e}")
        else:
            try:
                with open(self.presets_file, "w", encoding="utf-8") as f:
                    json.dump(presets, f, indent=4, ensure_ascii=False)
            except Exception:
                pass
        return presets

    def _save_presets_to_disk(self):
        try:
            with open(self.presets_file, "w", encoding="utf-8") as f:
                json.dump(self.all_presets, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Fehler beim Speichern von presets.json: {e}")

    def _load_app_settings(self):
        defaults = {"manual_latency_ms": 70.0, "last_media_dir": "", "last_audio_dir": "",
                    "last_output_dir": "", "last_project_dir": ""}
        try:
            if os.path.exists(self.app_settings_file):
                with open(self.app_settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    defaults.update({k: v for k, v in data.items() if k in defaults})
        except Exception as e:
            print(f"Fehler beim Laden von app_settings.json: {e}")
        return defaults

    def _save_app_settings(self):
        try:
            with open(self.app_settings_file, "w", encoding="utf-8") as f:
                json.dump(self.app_settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Fehler beim Speichern von app_settings.json: {e}")

    def get_manual_latency_ms(self):
        try:
            return float(self.app_settings.get("manual_latency_ms", 70.0))
        except Exception:
            return 70.0

    def set_manual_latency_ms(self, value_ms):
        self.app_settings["manual_latency_ms"] = float(max(0.0, min(600.0, value_ms)))
        self._save_app_settings()

    def _last_dir(self, key):
        d = self.app_settings.get(key) or ""
        return d if d and os.path.isdir(d) else ""

    def _remember_dir(self, key, path):
        if not path:
            return
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if folder and self.app_settings.get(key) != folder:
            self.app_settings[key] = folder
            self._save_app_settings()

    def _suggest_output_name(self, suffix=""):
        base = "edit"
        if self.audio_path:
            base = os.path.splitext(os.path.basename(self.audio_path))[0][:40].strip() or "edit"
        return f"{base}_edit{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    def _ask_output_path(self, title, suffix=""):
        initial_dir = self._last_dir("last_output_dir") or (
            os.path.dirname(self.audio_path) if self.audio_path else ""
        )
        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=initial_dir or None,
            initialfile=self._suggest_output_name(suffix),
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if path:
            self._remember_dir("last_output_dir", path)
        return path

    # =====================================================================
    # Shortcuts, Drag & Drop, Fenster
    # =====================================================================
    def _window_scaling(self, win=None):
        try:
            return float(ctk.ScalingTracker.get_window_scaling(win or self)) or 1.0
        except Exception:
            return 1.0

    def _fit_main_window(self):
        """Hauptfenster an den Bildschirm anpassen (auch bei 125/150 % Skalierung)."""
        scale = self._window_scaling()
        screen_w = self.winfo_screenwidth() / scale
        screen_h = self.winfo_screenheight() / scale
        w = int(min(1460, screen_w * 0.94))
        h = int(min(940, screen_h * 0.88))
        self.minsize(min(1000, w), min(640, h))
        x = int(max(0, (self.winfo_screenwidth() - w * scale) / 2))
        y = int(max(0, (self.winfo_screenheight() - h * scale) / 2 - 20 * scale))
        self.geometry(f"{w}x{h}+{x}+{y}")
        if screen_w < 1100 or screen_h < 700:
            try:
                self.state("zoomed")
            except Exception:
                pass

    def _bind_shortcuts(self):
        def guard(fn):
            def _handler(event=None):
                fn()
                return "break"
            return _handler

        self.bind("<Control-z>", guard(self.undo_action))
        self.bind("<Control-y>", guard(self.redo_action))
        self.bind("<Control-Z>", guard(self.redo_action))
        self.bind("<Control-s>", guard(self.save_project))
        self.bind("<Control-o>", guard(self.load_project))
        self.bind("<Control-Return>", guard(self.start_generation))
        self.bind("<Control-p>", guard(self.toggle_audio_preview))

    def _setup_drag_and_drop(self):
        if not self.dnd_enabled:
            return
        targets = [self, self.media_list_frame, self.wave_canvas, self.column_left, self.column_right]
        for widget in targets:
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_files_dropped)
            except Exception as e:
                print(f"[Drag & Drop] Ziel nicht registrierbar: {e}")

    def _on_files_dropped(self, event):
        try:
            paths = list(self.tk.splitlist(event.data))
        except Exception:
            paths = [event.data]
        self.handle_dropped_paths(paths)
        return getattr(event, "action", None)

    def handle_dropped_paths(self, paths):
        """Dateien sortieren: Videos/Bilder -> Medien, Audio -> Song, .aep -> Projekt."""
        media, audio, project = [], None, None
        for p in paths:
            if os.path.isdir(p):
                for name in sorted(os.listdir(p)):
                    full = os.path.join(p, name)
                    if os.path.splitext(name)[1].lower() in VIDEO_EXTS + IMAGE_EXTS:
                        media.append(full)
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext in VIDEO_EXTS + IMAGE_EXTS:
                media.append(p)
            elif ext in AUDIO_EXTS and audio is None:
                audio = p
            elif ext in PROJECT_EXTS and project is None:
                project = p
        if project:
            self.load_project(project)
            return
        if media:
            self.add_media_paths(media)
        if audio:
            self._flush_history()
            if self.load_audio_into_ui(audio):
                self.note_change()
        if not media and not audio:
            self.set_status("Keine passenden Dateien im Drop (Video, Bild, Audio oder .aep).", None)

    def _on_app_close(self):
        self.is_cancelled = True
        try:
            if self.audio_ok:
                pygame.mixer.music.stop()
                pygame.mixer.stop()
        except Exception:
            pass
        if getattr(self, "current_preview_file", None):
            try:
                os.remove(self.current_preview_file)
            except Exception:
                pass
        self.destroy()

    # =====================================================================
    # UI Building
    # =====================================================================
    def _build_ui(self):
        """
        Layout: Kopfleiste -> Profilleiste -> zwei scrollende Spalten ->
        fixierte Render-Leiste am unteren Rand (immer sichtbar).
        """
        self._fit_main_window()

        self.root_container = ctk.CTkFrame(self, fg_color="transparent")
        self.root_container.pack(fill="both", expand=True, padx=14, pady=(12, 10))

        self._build_header(self.root_container)
        self._build_profile_bar(self.root_container)

        body = ctk.CTkFrame(self.root_container, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1, uniform="cols")
        body.grid_columnconfigure(1, weight=1, uniform="cols")
        body.grid_rowconfigure(0, weight=1)

        self.column_left = ctk.CTkScrollableFrame(
            body, corner_radius=12, scrollbar_button_color="#334155",
            fg_color="#0f172a"
        )
        self.column_left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        self.column_right = ctk.CTkScrollableFrame(
            body, corner_radius=12, scrollbar_button_color="#334155",
            fg_color="#0f172a"
        )
        self.column_right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        self._build_media_section(self.column_left)
        self._build_audio_section(self.column_left)
        self._build_sync_section(self.column_left)
        self._build_format_section(self.column_left)

        self._build_audio_mix_section(self.column_right)
        self._build_effects_section(self.column_right)
        self._build_overlay_section(self.column_right)

        self._build_footer(self.root_container)

    # -----------------------------------------------------------------
    # Kopfleiste
    # -----------------------------------------------------------------
    def _build_header(self, parent):
        header = ctk.CTkFrame(parent, corner_radius=12, fg_color="#111c33")
        header.pack(fill="x", pady=(0, 9))

        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=16, pady=10)

        ctk.CTkLabel(
            left, text="Auto Beat-Sync Edit Maker",
            font=("Arial", 22, "bold")
        ).pack(anchor="w")

        ctk.CTkLabel(
            left,
            text="Clips + Song rein  ->  Bereich waehlen  ->  Rendern."
                 + ("  Dateien einfach ins Fenster ziehen." if self.dnd_enabled else ""),
            text_color="#38bdf8", font=("Arial", 11)
        ).pack(anchor="w")

        right = ctk.CTkFrame(header, fg_color="transparent")
        right.pack(side="right", padx=16, pady=10)

        ctk.CTkLabel(
            right, text="Strg+Z / Strg+Y  |  Strg+S Projekt  |  Strg+Enter Rendern",
            font=("Arial", 10), text_color="#94a3b8"
        ).pack(anchor="e", pady=(0, 3))

        hist_row = ctk.CTkFrame(right, fg_color="transparent")
        hist_row.pack(anchor="e")

        self.btn_undo = ctk.CTkButton(
            hist_row, text="Undo", command=self.undo_action, width=80, height=28,
            fg_color="#334155", hover_color="#475569"
        )
        self.btn_undo.pack(side="left", padx=3)

        self.btn_redo = ctk.CTkButton(
            hist_row, text="Redo", command=self.redo_action, width=80, height=28,
            fg_color="#334155", hover_color="#475569"
        )
        self.btn_redo.pack(side="left", padx=3)

    # -----------------------------------------------------------------
    # Profil-/Preset-Leiste (volle Breite, immer sichtbar)
    # -----------------------------------------------------------------
    def _build_profile_bar(self, parent):
        profile_frame = ctk.CTkFrame(parent, corner_radius=12, fg_color="#111c33")
        profile_frame.pack(fill="x", pady=(0, 9))

        row = ctk.CTkFrame(profile_frame, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=8)

        ctk.CTkLabel(row, text="Profil", font=("Arial", 13, "bold")).pack(side="left", padx=(0, 8))

        self.preset_names = list(self.all_presets.keys())
        self.profile_var = ctk.StringVar(value="Profil wählen...")
        self.combo_profile = ctk.CTkOptionMenu(
            row, values=self.preset_names, variable=self.profile_var,
            command=self.on_profile_selected, width=260, dynamic_resizing=False
        )
        self.combo_profile.pack(side="left", padx=(0, 8))

        ctk.CTkLabel(
            row, text="Projekt", font=("Arial", 13, "bold")
        ).pack(side="left", padx=(22, 8))

        self.btn_save_project = ctk.CTkButton(
            row, text="Speichern", command=self.save_project,
            fg_color="#1f8a4c", hover_color="#176b3b", width=110, height=30
        )
        self.btn_save_project.pack(side="left", padx=3)

        self.btn_load_project = ctk.CTkButton(
            row, text="Laden", command=self.load_project,
            fg_color="#1f6f8a", hover_color="#165266", width=90, height=30
        )
        self.btn_load_project.pack(side="left", padx=3)

        ctk.CTkLabel(
            row, text="Preset", font=("Arial", 13, "bold")
        ).pack(side="left", padx=(22, 8))

        self.btn_save_profile = ctk.CTkButton(
            row, text="Speichern", command=self.save_current_as_profile,
            fg_color="#1f8a4c", hover_color="#176b3b", width=110, height=30
        )
        self.btn_save_profile.pack(side="left", padx=3)

        self.btn_load_json = ctk.CTkButton(
            row, text="JSON laden", command=self.import_profile_json,
            fg_color="#2b719e", hover_color="#1f5375", width=100, height=30
        )
        self.btn_load_json.pack(side="left", padx=3)

        self.btn_export_json = ctk.CTkButton(
            row, text="Export", command=self.export_profile_json,
            fg_color="#444444", hover_color="#555555", width=80, height=30
        )
        self.btn_export_json.pack(side="left", padx=3)

        self.btn_delete_prof = ctk.CTkButton(
            row, text="Loeschen", command=self.delete_current_profile,
            fg_color="#8b1e1e", hover_color="#631414", width=90, height=30
        )
        self.btn_delete_prof.pack(side="left", padx=3)

    # -----------------------------------------------------------------
    # 1. Medien
    # -----------------------------------------------------------------
    def _build_media_section(self, parent):
        media_frame = ctk.CTkFrame(parent, corner_radius=12)
        media_frame.pack(fill="x", pady=(0, 9))

        header_media = ctk.CTkFrame(media_frame, fg_color="transparent")
        header_media.pack(fill="x", padx=14, pady=(10, 2))

        ctk.CTkLabel(
            header_media, text="1. Medien (Videos & Bilder)", font=("Arial", 14, "bold")
        ).pack(side="left")
        self.lbl_videos_count = ctk.CTkLabel(header_media, text="0 Dateien ausgewählt", text_color="gray")
        self.lbl_videos_count.pack(side="right")

        btn_row_media = ctk.CTkFrame(media_frame, fg_color="transparent")
        btn_row_media.pack(fill="x", padx=14, pady=4)

        self.btn_select_videos = ctk.CTkButton(
            btn_row_media, text="Clips / Bilder wählen", command=self.select_videos, width=170, height=32
        )
        self.btn_select_videos.pack(side="left", padx=(0, 8))

        self.btn_clear_videos = ctk.CTkButton(
            btn_row_media, text="Zurücksetzen", command=self.clear_videos,
            fg_color="#444444", hover_color="#555555", width=110, height=32
        )
        self.btn_clear_videos.pack(side="left", padx=(0, 10))

        self.shuffle_var = ctk.BooleanVar(value=True)
        self.chk_shuffle = ctk.CTkCheckBox(
            btn_row_media, text="Zufällige Reihenfolge", variable=self.shuffle_var
        )
        self.chk_shuffle.pack(side="left", padx=5)

        self.media_list_frame = ctk.CTkScrollableFrame(
            media_frame, height=120, fg_color="#0b1220", corner_radius=8
        )
        self.media_list_frame.pack(fill="x", padx=14, pady=(2, 10))
        self._media_rows = []
        self._update_media_listbox()

    # -----------------------------------------------------------------
    # 2. Musik
    # -----------------------------------------------------------------
    def _build_audio_section(self, parent):
        audio_frame = ctk.CTkFrame(parent, corner_radius=12)
        audio_frame.pack(fill="x", pady=(0, 9))

        header_audio = ctk.CTkFrame(audio_frame, fg_color="transparent")
        header_audio.pack(fill="x", padx=14, pady=(10, 2))

        ctk.CTkLabel(
            header_audio, text="2. Musik (Beat & Song)", font=("Arial", 14, "bold")
        ).pack(side="left")
        self.lbl_audio_status = ctk.CTkLabel(header_audio, text="Kein Song gewählt", text_color="gray")
        self.lbl_audio_status.pack(side="right")

        btn_row_audio = ctk.CTkFrame(audio_frame, fg_color="transparent")
        btn_row_audio.pack(fill="x", padx=14, pady=4)

        self.btn_select_audio = ctk.CTkButton(
            btn_row_audio, text="Song auswählen", command=self.select_audio, width=160, height=32
        )
        self.btn_select_audio.pack(side="left", padx=(0, 8))

        self.btn_audio_preview = ctk.CTkButton(
            btn_row_audio, text="Vorhören", command=self.toggle_audio_preview,
            fg_color="#2b719e", hover_color="#1f5375", width=120, height=32, state="disabled"
        )
        self.btn_audio_preview.pack(side="left", padx=5)

        slider_frame = ctk.CTkFrame(audio_frame, fg_color="transparent")
        slider_frame.pack(fill="x", padx=14, pady=(4, 4))

        start_row = ctk.CTkFrame(slider_frame, fg_color="transparent")
        start_row.pack(fill="x", pady=2)
        self.lbl_start = ctk.CTkLabel(start_row, text="Start: 0.0s", width=90, anchor="w", font=("Arial", 12))
        self.lbl_start.pack(side="left")
        self.slider_start = ctk.CTkSlider(start_row, from_=0, to=100, command=self.update_start_lbl)
        self.slider_start.set(0)
        self.slider_start.pack(side="right", fill="x", expand=True, padx=5)

        end_row = ctk.CTkFrame(slider_frame, fg_color="transparent")
        end_row.pack(fill="x", pady=2)
        self.lbl_end = ctk.CTkLabel(end_row, text="Ende: 0.0s", width=90, anchor="w", font=("Arial", 12))
        self.lbl_end.pack(side="left")
        self.slider_end = ctk.CTkSlider(end_row, from_=0, to=100, command=self.update_end_lbl)
        self.slider_end.set(0)
        self.slider_end.pack(side="right", fill="x", expand=True, padx=5)

        waveform_wrap = ctk.CTkFrame(audio_frame, fg_color="transparent")
        waveform_wrap.pack(fill="x", padx=14, pady=(2, 4))

        # Hinweise in eigener Zeile - frueher teilten sie sich die Zeile mit
        # der Wellenform, die dadurch nur ein schmaler Streifen rechts war.
        hint_row = ctk.CTkFrame(waveform_wrap, fg_color="transparent")
        hint_row.pack(fill="x")

        self.lbl_waveform_hint = ctk.CTkLabel(
            hint_row, text="Links: Start | Rechts: Ende | Mausrad: Zoom | Mitte: Verschieben",
            font=("Arial", 10), text_color="#94a3b8"
        )
        self.lbl_waveform_hint.pack(side="left")

        self.lbl_playback_time = ctk.CTkLabel(
            hint_row, text="", font=("Consolas", 10, "bold"), text_color="#facc15"
        )
        self.lbl_playback_time.pack(side="right")

        self.wave_canvas = Canvas(
            waveform_wrap, height=int(96 * self._window_scaling()), bg="#0b1220",
            highlightthickness=1, highlightbackground="#1e293b", cursor="hand2"
        )
        self.wave_canvas.pack(fill="x", pady=(2, 8))
        self.wave_canvas.bind("<Configure>", lambda e: self.draw_waveform())
        self.wave_canvas.bind("<Button-1>", self.on_waveform_click)
        self.wave_canvas.bind("<Button-3>", self.on_waveform_right_click)
        self.wave_canvas.bind("<MouseWheel>", self.on_waveform_wheel)
        self.wave_canvas.bind("<ButtonPress-2>", self.on_waveform_pan_start)
        self.wave_canvas.bind("<B2-Motion>", self.on_waveform_pan)
        self.wave_canvas.bind("<Double-Button-1>", self.on_waveform_reset_view)
        self.wave_env = None
        self.wave_hop = 0.0
        self.wave_beats = []
        self.wave_drop = 0.0
        self.wave_playhead = None
        self.wave_view = (0.0, 1.0)   # von load_audio_into_ui auf die echte Dauer gesetzt
        self._pan_last_x = 0
        self.drop_candidates = []
        self.drop_scan_offset = 0.0
        self._candidate_win = None

    # -----------------------------------------------------------------
    # 3. Beat-Sync & Timing
    # -----------------------------------------------------------------
    def _build_sync_section(self, parent):
        sync_frame = ctk.CTkFrame(parent, corner_radius=12)
        sync_frame.pack(fill="x", pady=(0, 9))

        ctk.CTkLabel(
            sync_frame, text="3. Beat-Sync & Multi-Band Timing", font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=14, pady=(10, 3))

        mode_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        mode_row.pack(fill="x", padx=14, pady=2)

        self.mode_var = ctk.StringVar(value="auto")
        ctk.CTkRadioButton(
            mode_row, text="Auto Takt (Kick & Snare)", variable=self.mode_var, value="auto"
        ).pack(side="left", padx=(0, 14))
        ctk.CTkRadioButton(
            mode_row, text="Dynamisch (Alle Onsets)", variable=self.mode_var, value="onsets"
        ).pack(side="left", padx=(0, 14))
        ctk.CTkRadioButton(
            mode_row, text="Manuell (Takt-Magnet)", variable=self.mode_var, value="manual"
        ).pack(side="left")

        pace_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        pace_row.pack(fill="x", padx=14, pady=2)

        ctk.CTkLabel(pace_row, text="Schnitt-Tempo:").pack(side="left", padx=(0, 8))
        self.pace_var = ctk.StringVar(value=PACE_OPTIONS[0])
        self.combo_pace = ctk.CTkOptionMenu(
            pace_row, values=PACE_OPTIONS, variable=self.pace_var, width=200
        )
        self.combo_pace.pack(side="left", padx=(0, 14))

        ctk.CTkLabel(pace_row, text="Framerate:").pack(side="left", padx=(0, 6))
        self.fps_var = ctk.IntVar(value=60)
        ctk.CTkRadioButton(pace_row, text="60", variable=self.fps_var, value=60).pack(side="left", padx=(0, 8))
        ctk.CTkRadioButton(pace_row, text="30", variable=self.fps_var, value=30).pack(side="left")

        opt_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        opt_row.pack(fill="x", padx=14, pady=(2, 4))

        self.ai_audio_analysis_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opt_row, text="KI-Struktur (ML-Cluster)",
            variable=self.ai_audio_analysis_var, font=("Arial", 11)
        ).pack(side="left", padx=(0, 16))

        self.snap_jitter_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opt_row, text="Jitter-Korrektur", variable=self.snap_jitter_var, font=("Arial", 11)
        ).pack(side="left")

        preroll_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        preroll_row.pack(fill="x", padx=14, pady=2)

        ctk.CTkLabel(preroll_row, text="Drop-Pre-Roll / Intro:").pack(side="left", padx=(0, 8))
        self.preroll_var = ctk.StringVar(value=PREROLL_OPTIONS[4])
        ctk.CTkOptionMenu(
            preroll_row, values=PREROLL_OPTIONS, variable=self.preroll_var, width=215
        ).pack(side="left")
        ctk.CTkLabel(
            preroll_row, text="nur Auto/Dynamisch", font=("Arial", 10), text_color="#64748b"
        ).pack(side="left", padx=(8, 0))

        # --- Gespeicherte manuelle Cuts ------------------------------
        # Die Cuts aus dem Aufnahme-Review landen hier und werden von
        # Rendern, Schnell-Vorschau und Batch-Export gemeinsam genutzt.
        cuts_frame = ctk.CTkFrame(sync_frame, fg_color="#0d1526", corner_radius=8)
        cuts_frame.pack(fill="x", padx=14, pady=(8, 4))

        cuts_row = ctk.CTkFrame(cuts_frame, fg_color="transparent")
        cuts_row.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(
            cuts_row, text="Manuelle Cuts", font=("Arial", 12, "bold")
        ).pack(side="left", padx=(0, 10))

        self.use_manual_cuts_var = ctk.BooleanVar(value=False)
        self.chk_manual_cuts = ctk.CTkCheckBox(
            cuts_row, text="Beim Rendern verwenden", variable=self.use_manual_cuts_var,
            font=("Arial", 11), command=self._on_manual_cuts_toggle
        )
        self.chk_manual_cuts.pack(side="left", padx=(0, 10))

        cuts_btn_row = ctk.CTkFrame(cuts_frame, fg_color="transparent")
        cuts_btn_row.pack(fill="x", padx=10, pady=(4, 2))

        self.btn_record = ctk.CTkButton(
            cuts_btn_row, text="Jetzt aufnehmen", command=self.start_manual_recording,
            fg_color="#b45309", hover_color="#92400e", width=130, height=30
        )
        self.btn_record.pack(side="left", padx=(0, 6))

        self.btn_manual_cuts_view = ctk.CTkButton(
            cuts_btn_row, text="Cuts ansehen", command=self.open_saved_cuts_review,
            fg_color="#2b719e", hover_color="#1f5375", width=110, height=30,
            state="disabled"
        )
        self.btn_manual_cuts_view.pack(side="left", padx=6)

        self.btn_manual_cuts_range = ctk.CTkButton(
            cuts_btn_row, text="Bereich laden", command=self.apply_manual_cuts_range,
            fg_color="#4b5563", hover_color="#374151", width=110, height=30,
            state="disabled"
        )
        self.btn_manual_cuts_range.pack(side="left", padx=6)

        self.btn_manual_cuts_clear = ctk.CTkButton(
            cuts_btn_row, text="Verwerfen", command=lambda: self.clear_manual_cuts(confirm=True),
            fg_color="#8b1e1e", hover_color="#631414", width=90, height=30,
            state="disabled"
        )
        self.btn_manual_cuts_clear.pack(side="left", padx=6)

        self.lbl_manual_cuts = ctk.CTkLabel(
            cuts_frame,
            text="Keine Cuts gespeichert - 'Jetzt aufnehmen' tippt die Schnitte live zur Musik.",
            font=("Arial", 10), text_color="#64748b", anchor="w", justify="left", wraplength=560
        )
        self.lbl_manual_cuts.pack(fill="x", padx=10, pady=(0, 8))

        candidate_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        candidate_row.pack(fill="x", padx=14, pady=(2, 4))

        self.btn_candidates = ctk.CTkButton(
            candidate_row, text="Drop-Kandidaten suchen", width=190, height=32,
            fg_color="#db2777", hover_color="#9d174d",
            command=self.scan_drop_candidates_ui
        )
        self.btn_candidates.pack(side="left", padx=(0, 8))

        self.lbl_candidate_hint = ctk.CTkLabel(
            candidate_row, text="KI durchsucht den Auswahlbereich nach allen Drops",
            font=("Arial", 10), text_color="#94a3b8"
        )
        self.lbl_candidate_hint.pack(side="left")

        best_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        best_row.pack(fill="x", padx=14, pady=(2, 10))

        ctk.CTkLabel(best_row, text="Best-Moment-Länge:").pack(side="left", padx=(0, 8))
        self.best_moment_len_var = ctk.StringVar(value="30s")
        ctk.CTkOptionMenu(
            best_row, values=["15s", "30s", "60s", "90s"], variable=self.best_moment_len_var, width=70
        ).pack(side="left", padx=(0, 8))

        self.btn_best_moment = ctk.CTkButton(
            best_row, text="Besten Hook finden", width=160, height=32,
            fg_color="#7c3aed", hover_color="#5b21b6",
            command=self.find_best_moment_ui
        )
        self.btn_best_moment.pack(side="left")

    # -----------------------------------------------------------------
    # 4. Format, Auflösung, Qualität
    # -----------------------------------------------------------------
    def _build_format_section(self, parent):
        format_frame = ctk.CTkFrame(parent, corner_radius=12)
        format_frame.pack(fill="x", pady=(0, 9))

        ctk.CTkLabel(
            format_frame, text="4. Format, Auflösung & Qualität", font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=14, pady=(10, 3))

        row_fmt = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_fmt.pack(fill="x", padx=14, pady=2)

        ctk.CTkLabel(row_fmt, text="Format:").pack(side="left", padx=(0, 8))
        self.orient_var = ctk.StringVar(value="portrait")
        ctk.CTkRadioButton(row_fmt, text="Original", variable=self.orient_var, value="original").pack(side="left", padx=(0, 10))
        ctk.CTkRadioButton(row_fmt, text="Hochkant 9:16", variable=self.orient_var, value="portrait").pack(side="left", padx=(0, 10))
        ctk.CTkRadioButton(row_fmt, text="Quer 16:9", variable=self.orient_var, value="landscape").pack(side="left")

        row_scale = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_scale.pack(fill="x", padx=14, pady=2)

        ctk.CTkLabel(row_scale, text="Einpassung:").pack(side="left", padx=(0, 8))
        self.scale_mode_var = ctk.StringVar(value="fit")
        ctk.CTkRadioButton(row_scale, text="Ganzes Bild (Fit)", variable=self.scale_mode_var, value="fit").pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(row_scale, text="Vollbild (Crop)", variable=self.scale_mode_var, value="fill").pack(side="left")

        row_quality = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_quality.pack(fill="x", padx=14, pady=(2, 2))

        ctk.CTkLabel(row_quality, text="Qualität:").pack(side="left", padx=(0, 8))
        self.quality_var = ctk.StringVar(value=QUALITY_OPTIONS[0])
        ctk.CTkOptionMenu(
            row_quality, values=QUALITY_OPTIONS, variable=self.quality_var, width=215
        ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(row_quality, text="Auflösung:").pack(side="left", padx=(0, 8))
        self.res_var = ctk.StringVar(value=RES_OPTIONS[0])
        self.combo_res = ctk.CTkOptionMenu(
            row_quality, values=RES_OPTIONS, variable=self.res_var, width=185,
            command=lambda _v: self.refresh_resolution_warning()
        )
        self.combo_res.pack(side="left", padx=(0, 10))

        # Upscaling erfindet Bildpunkte und sieht sofort weich aus. Der Hinweis
        # erscheint nur, wenn eine Aufloesung gewaehlt ist, die groesser ist als
        # die hoechste Quellaufloesung der aktuell geladenen Videos.
        self.lbl_res_warn = ctk.CTkLabel(
            format_frame, text="", font=("Arial", 10), text_color="#f59e0b", anchor="w"
        )
        self.lbl_res_warn.pack(fill="x", padx=14)

        row_reframe = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_reframe.pack(fill="x", padx=14, pady=(4, 10))

        self.ai_reframe_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            row_reframe, text="KI-Auto-Reframe (Gesicht im 9:16-Crop halten)",
            variable=self.ai_reframe_var
        ).pack(side="left")

    # -----------------------------------------------------------------
    # 5. Audio-Mixing
    # -----------------------------------------------------------------
    def _build_audio_mix_section(self, parent):
        audio_opt_frame = ctk.CTkFrame(parent, corner_radius=12)
        audio_opt_frame.pack(fill="x", pady=(0, 9))

        ctk.CTkLabel(
            audio_opt_frame, text="5. Audio-Mixing & Ausblenden", font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=14, pady=(10, 3))

        row_mix = ctk.CTkFrame(audio_opt_frame, fg_color="transparent")
        row_mix.pack(fill="x", padx=14, pady=(2, 2))

        ctk.CTkLabel(row_mix, text="Original Clip-Sound:").pack(side="left", padx=(0, 8))
        self.clip_audio_var = ctk.StringVar(value=CLIP_AUDIO_OPTIONS[0])
        ctk.CTkOptionMenu(
            row_mix, values=CLIP_AUDIO_OPTIONS, variable=self.clip_audio_var, width=230
        ).pack(side="left")

        row_audio_flags = ctk.CTkFrame(audio_opt_frame, fg_color="transparent")
        row_audio_flags.pack(fill="x", padx=14, pady=(2, 10))

        self.fade_out_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row_audio_flags, text="Fade-Out am Ende (0.8s)",
            variable=self.fade_out_var
        ).pack(side="left", padx=(0, 18))

        self.auto_loudness_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row_audio_flags, text="Auto-Loudness (EBU R128, -14 LUFS)",
            variable=self.auto_loudness_var
        ).pack(side="left")

    # -----------------------------------------------------------------
    # 6. Impact-Effekte
    # -----------------------------------------------------------------
    def _build_effects_section(self, parent):
        effects_frame = ctk.CTkFrame(parent, corner_radius=12)
        effects_frame.pack(fill="x", pady=(0, 9))

        ctk.CTkLabel(
            effects_frame, text="6. Impact-Effekte & Cinematic Look",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=14, pady=(10, 0))

        ctk.CTkLabel(
            effects_frame,
            text="Das Effekt-Set bestimmt Staerke und Haeufigkeit. Die Haekchen schalten einzelne Effekte ab.",
            font=("Arial", 10), text_color="#64748b", anchor="w"
        ).pack(fill="x", padx=14, pady=(0, 4))

        cadence_row = ctk.CTkFrame(effects_frame, fg_color="transparent")
        cadence_row.pack(fill="x", padx=14, pady=2)

        ctk.CTkLabel(cadence_row, text="Effekt-Set:").pack(side="left", padx=(0, 8))
        self.effect_set_var = ctk.StringVar(value="CapCut Kinetic (Empfohlen)")
        ctk.CTkOptionMenu(
            cadence_row, values=list(EFFECT_SETS.keys()),
            variable=self.effect_set_var, width=225
        ).pack(side="left")

        ctk.CTkLabel(cadence_row, text="Akzente:").pack(side="left", padx=(16, 8))
        self.fx_cadence_var = ctk.StringVar(value=CADENCE_OPTIONS[1])
        ctk.CTkOptionMenu(
            cadence_row, values=CADENCE_OPTIONS, variable=self.fx_cadence_var, width=185
        ).pack(side="left")

        intensity_row = ctk.CTkFrame(effects_frame, fg_color="transparent")
        intensity_row.pack(fill="x", padx=14, pady=(6, 2))

        self.lbl_fx_intensity = ctk.CTkLabel(
            intensity_row, text="Effekt-Intensität: 100%", width=170, anchor="w", font=("Arial", 11)
        )
        self.lbl_fx_intensity.pack(side="left", padx=(0, 10))

        self.slider_fx_intensity = ctk.CTkSlider(
            intensity_row, from_=20, to=200,
            command=lambda v: (self.lbl_fx_intensity.configure(text=f"Effekt-Intensität: {int(v)}%"), self.note_change())
        )
        self.slider_fx_intensity.set(100)
        self.slider_fx_intensity.pack(side="left", fill="x", expand=True)

        eff_grid = ctk.CTkFrame(effects_frame, fg_color="transparent")
        eff_grid.pack(fill="x", padx=14, pady=(6, 10))
        for col in range(3):
            eff_grid.grid_columnconfigure(col, weight=1)

        self.fx_punch_zoom = ctk.BooleanVar(value=True)
        self.fx_bloom_flash = ctk.BooleanVar(value=True)
        self.fx_shake = ctk.BooleanVar(value=True)
        self.fx_speedramp = ctk.BooleanVar(value=True)
        self.fx_zoom_blur = ctk.BooleanVar(value=True)
        self.fx_collage = ctk.BooleanVar(value=True)
        self.fx_spin = ctk.BooleanVar(value=True)
        self.fx_cinematic_lut = ctk.BooleanVar(value=True)
        self.fx_vignette = ctk.BooleanVar(value=True)
        self.fx_grain = ctk.BooleanVar(value=True)

        boxes = [
            (self.fx_punch_zoom, "Punch In/Out + Whip"),
            (self.fx_bloom_flash, "Flash auf Akzenten"),
            (self.fx_shake, "Shake auf Akzenten"),
            (self.fx_zoom_blur, "Zoom-Blur am Drop"),
            (self.fx_speedramp, "Speed-Ramp"),
            (self.fx_collage, "Collage (Raster)"),
            (self.fx_spin, "Spin (360°)"),
            (self.fx_cinematic_lut, "Cinematic LUT"),
            (self.fx_vignette, "Vignette"),
            (self.fx_grain, "Filmkorn"),
        ]
        for idx, (var, label) in enumerate(boxes):
            ctk.CTkCheckBox(
                eff_grid, text=label, variable=var, font=("Arial", 11)
            ).grid(row=idx // 3, column=idx % 3, sticky="w", padx=(0, 10), pady=2)

    # -----------------------------------------------------------------
    # 7. Spinning Overlay
    # -----------------------------------------------------------------
    def _build_overlay_section(self, parent):
        overlay_frame = ctk.CTkFrame(parent, corner_radius=12)
        overlay_frame.pack(fill="x", pady=(0, 9))

        overlay_header = ctk.CTkFrame(overlay_frame, fg_color="transparent")
        overlay_header.pack(fill="x", padx=14, pady=(10, 2))

        self.enable_overlay_var = ctk.BooleanVar(value=False)
        self.chk_overlay = ctk.CTkCheckBox(
            overlay_header, text="7. Spinning Bild-Overlay", variable=self.enable_overlay_var,
            command=self._toggle_overlay_ui, font=("Arial", 14, "bold")
        )
        self.chk_overlay.pack(side="left")

        self.overlay_options_frame = ctk.CTkFrame(overlay_frame, fg_color="transparent")

        row_ov1 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov1.pack(fill="x", padx=14, pady=4)

        self.btn_select_overlay = ctk.CTkButton(
            row_ov1, text="Overlay-Bild wählen", command=self.select_overlay_image, width=170, height=30
        )
        self.btn_select_overlay.pack(side="left", padx=(0, 10))

        self.lbl_overlay_name = ctk.CTkLabel(row_ov1, text="Kein Bild ausgewählt", text_color="gray")
        self.lbl_overlay_name.pack(side="left")

        row_ov2 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov2.pack(fill="x", padx=14, pady=4)

        ctk.CTkLabel(row_ov2, text="Start:").pack(side="left", padx=(0, 8))
        self.overlay_start_mode = ctk.StringVar(value="drop")
        for text, value in (("Am erkannten Drop", "drop"), ("Ab Sekunde", "custom"), ("Von Anfang an", "start")):
            ctk.CTkRadioButton(
                row_ov2, text=text, variable=self.overlay_start_mode, value=value
            ).pack(side="left", padx=(0, 10))

        self.row_ov_slider = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        self.row_ov_slider.pack(fill="x", padx=14, pady=2)

        self.lbl_overlay_sec = ctk.CTkLabel(
            self.row_ov_slider, text="Ab Sekunde: 0.0s", width=120, anchor="w", font=("Arial", 11)
        )
        self.lbl_overlay_sec.pack(side="left")

        self.btn_detect_drop = ctk.CTkButton(
            self.row_ov_slider, text="Drop finden", width=110, height=28,
            fg_color="#334155", hover_color="#475569", command=self.auto_detect_drop_ui
        )
        self.btn_detect_drop.pack(side="right", padx=(8, 0))

        # Sekunden relativ zum gewaehlten Musik-Start (0 = Anfang des Edits)
        self.slider_overlay_sec = ctk.CTkSlider(
            self.row_ov_slider, from_=0, to=120,
            command=lambda v: (self.lbl_overlay_sec.configure(text=f"Ab Sekunde: {v:.1f}s"),
                               self.overlay_start_mode.set("custom"), self.note_change())
        )
        self.slider_overlay_sec.set(0.0)
        self.slider_overlay_sec.pack(side="right", fill="x", expand=True, padx=8)

        row_ov3 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov3.pack(fill="x", padx=14, pady=4)

        ctk.CTkLabel(row_ov3, text="Platzierung:").pack(side="left", padx=(0, 8))
        self.overlay_corner_var = ctk.StringVar(value=OVERLAY_CORNER_OPTIONS[0])
        ctk.CTkOptionMenu(
            row_ov3, values=OVERLAY_CORNER_OPTIONS, variable=self.overlay_corner_var, width=215
        ).pack(side="left")

        row_ov4 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov4.pack(fill="x", padx=14, pady=4)

        ctk.CTkLabel(row_ov4, text="Größe:").pack(side="left", padx=(0, 8))
        self.overlay_size_var = ctk.StringVar(value=OVERLAY_SIZE_OPTIONS[1])
        ctk.CTkOptionMenu(
            row_ov4, values=OVERLAY_SIZE_OPTIONS, variable=self.overlay_size_var, width=210
        ).pack(side="left", padx=(0, 14))

        self.overlay_circle_cut_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row_ov4, text="Kreis-Zuschnitt", variable=self.overlay_circle_cut_var
        ).pack(side="left")

        row_ov5 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov5.pack(fill="x", padx=14, pady=(4, 10))

        speed_col = ctk.CTkFrame(row_ov5, fg_color="transparent")
        speed_col.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.lbl_ov_speed = ctk.CTkLabel(speed_col, text="Dreh-Speed: 1.2x", anchor="w", font=("Arial", 11))
        self.lbl_ov_speed.pack(fill="x")
        self.slider_ov_speed = ctk.CTkSlider(
            speed_col, from_=0.2, to=4.0,
            command=lambda v: (self.lbl_ov_speed.configure(text=f"Dreh-Speed: {v:.1f}x"), self.note_change())
        )
        self.slider_ov_speed.set(1.2)
        self.slider_ov_speed.pack(fill="x", pady=2)

        opacity_col = ctk.CTkFrame(row_ov5, fg_color="transparent")
        opacity_col.pack(side="right", fill="x", expand=True, padx=(8, 0))
        self.lbl_ov_opacity = ctk.CTkLabel(opacity_col, text="Deckkraft: 100%", anchor="w", font=("Arial", 11))
        self.lbl_ov_opacity.pack(fill="x")
        self.slider_ov_opacity = ctk.CTkSlider(
            opacity_col, from_=10, to=100,
            command=lambda v: (self.lbl_ov_opacity.configure(text=f"Deckkraft: {int(v)}%"), self.note_change())
        )
        self.slider_ov_opacity.set(100.0)
        self.slider_ov_opacity.pack(fill="x", pady=2)

    # -----------------------------------------------------------------
    # Render-Leiste (fixiert am unteren Rand)
    # -----------------------------------------------------------------
    def _build_footer(self, parent):
        bottom_frame = ctk.CTkFrame(parent, corner_radius=12, fg_color="#111c33")
        bottom_frame.pack(fill="x", pady=(9, 0))

        btn_row_action = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        btn_row_action.pack(fill="x", padx=16, pady=(10, 4))

        self.btn_generate = ctk.CTkButton(
            btn_row_action, text="Video Rendern", command=self.start_generation,
            fg_color="#1f8a4c", hover_color="#176b3b", height=44, width=220,
            font=("Arial", 15, "bold")
        )
        self.btn_generate.pack(side="left", padx=(0, 8))

        self.btn_proxy = ctk.CTkButton(
            btn_row_action, text="Schnell-Vorschau", command=self.start_proxy_render,
            fg_color="#7c3aed", hover_color="#5b21b6", height=44, width=170,
            font=("Arial", 13, "bold")
        )
        self.btn_proxy.pack(side="left", padx=(0, 8))

        self.btn_batch = ctk.CTkButton(
            btn_row_action, text="Batch-Export", command=self.start_batch_export,
            fg_color="#b45309", hover_color="#92400e", height=44, width=150,
            font=("Arial", 13, "bold")
        )
        self.btn_batch.pack(side="left", padx=(0, 8))

        self.post_action_frame = ctk.CTkFrame(btn_row_action, fg_color="transparent")
        self.post_action_frame.pack_forget()
        self.btn_open_folder = ctk.CTkButton(
            self.post_action_frame, text="Öffnen", command=self.open_output_folder,
            fg_color="#2b719e", hover_color="#1f5375", width=120, height=44
        )
        self.btn_open_folder.pack(side="left", padx=(0, 8))

        self.btn_play_video = ctk.CTkButton(
            self.post_action_frame, text="Ansehen", command=self.play_output_video,
            fg_color="#2563eb", hover_color="#1d4ed8", width=140, height=44
        )
        self.btn_play_video.pack(side="left", padx=(0, 8))

        self.btn_cancel = ctk.CTkButton(
            btn_row_action, text="Abbrechen", command=self.cancel_generation,
            fg_color="#8b1e1e", hover_color="#631414", height=44, width=120,
            font=("Arial", 13, "bold"), state="disabled"
        )
        self.btn_cancel.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(bottom_frame, height=14)
        self.progress_bar.pack(fill="x", padx=16, pady=(6, 4))
        self.progress_bar.set(0.0)

        status_row = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        status_row.pack(fill="x", padx=16, pady=(0, 8))

        self.lbl_status = ctk.CTkLabel(status_row, text="Bereit für Schnitt", font=("Arial", 12), text_color="yellow")
        self.lbl_status.pack(side="left")

        self.lbl_eta = ctk.CTkLabel(status_row, text="", font=("Arial", 11), text_color="#94a3b8")
        self.lbl_eta.pack(side="left", padx=(16, 0))

        self.lbl_pct = ctk.CTkLabel(status_row, text="0%", font=("Arial", 13, "bold"))
        self.lbl_pct.pack(side="right")

    def _toggle_overlay_ui(self):

        if self.enable_overlay_var.get():
            self.overlay_options_frame.pack(fill="x", padx=5, pady=(0, 8))
        else:
            self.overlay_options_frame.pack_forget()

    def select_overlay_image(self):
        path = filedialog.askopenfilename(
            title="Wähle Bild für Spinning Overlay",
            filetypes=[("Bilder", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Alle Dateien", "*.*")]
        )
        if path:
            self.overlay_img_path = path
            self.enable_overlay_var.set(True)
            self._toggle_overlay_ui()
            self.lbl_overlay_name.configure(text=os.path.basename(path), text_color="#38bdf8")

    def auto_detect_drop_ui(self):
        if not self.audio_path or not os.path.exists(self.audio_path):
            self.show_error("Hinweis", "Bitte zuerst eine Audio-Datei auswählen.")
            return

        start_t = self.slider_start.get()
        end_t = self.slider_end.get()
        dur = max(1.0, end_t - start_t)

        def _worker():
            try:
                t0 = time.time()
                self.set_status("KI analysiert Musikstruktur & Drop-Cluster...", 0.3)
                y, sr = librosa.load(self.audio_path, offset=start_t, duration=dur, sr=22050)
                best_time = ai_detect_beat_drop(y, sr, len(y) / sr)
                elapsed = time.time() - t0

                def _apply():
                    clamped_t = max(0.0, min(dur, best_time))
                    self.slider_overlay_sec.set(clamped_t)
                    self.lbl_overlay_sec.configure(text=f"Ab Sekunde: {clamped_t:.1f}s")
                    self.overlay_start_mode.set("custom")
                    if best_time <= 0.05:
                        self.set_status("KI: Kein eindeutiger Drop gefunden - bitte manuell setzen.", 0.0)
                    else:
                        self.set_status(
                            f"KI: Beat-Drop erkannt bei {clamped_t:.1f}s ({elapsed:.1f}s Analyse)", 0.0
                        )

                self._ui_after(0, _apply)
            except Exception as e:
                self._ui_after(0, lambda: self.set_status(f"Fehler bei Drop-Erkennung: {e}", 0.0))

        threading.Thread(target=_worker, daemon=True).start()

    # =====================================================================
    # Preset & Profile Management
    # =====================================================================
    def on_profile_selected(self, choice):
        if choice in self.all_presets:
            self.apply_preset_dict(self.all_presets[choice])
            self.set_status(f"Profil '{choice}' geladen.", None)

    def _setting_vars(self):
        """Key -> Tk-Variable fuer alle Einstellungen, die Profile/Projekte speichern."""
        return {
            "mode": self.mode_var,
            "pace": self.pace_var,
            "orientation": self.orient_var,
            "scale_mode": self.scale_mode_var,
            "quality": self.quality_var,
            "resolution": self.res_var,
            "clip_audio": self.clip_audio_var,
            "fade_out": self.fade_out_var,
            "auto_loudness": self.auto_loudness_var,
            "snap_jitter": self.snap_jitter_var,
            "ai_audio_analysis": self.ai_audio_analysis_var,
            "ai_reframe": self.ai_reframe_var,
            "preroll": self.preroll_var,
            "effect_set": self.effect_set_var,
            "fx_cadence": self.fx_cadence_var,
            "fx_shake": self.fx_shake,
            "fx_bloom_flash": self.fx_bloom_flash,
            "fx_punch_zoom": self.fx_punch_zoom,
            "fx_speedramp": self.fx_speedramp,
            "fx_zoom_blur": self.fx_zoom_blur,
            "fx_cinematic_lut": self.fx_cinematic_lut,
            "fx_vignette": self.fx_vignette,
            "fx_grain": self.fx_grain,
            "fx_collage": self.fx_collage,
            "fx_spin": self.fx_spin,
            "enable_overlay": self.enable_overlay_var,
            "overlay_corner": self.overlay_corner_var,
            "overlay_size": self.overlay_size_var,
            "overlay_start_mode": self.overlay_start_mode,
            "overlay_circle": self.overlay_circle_cut_var,
            "shuffle": self.shuffle_var,
        }

    def get_current_settings_dict(self):
        data = {key: var.get() for key, var in self._setting_vars().items()}
        data["fps"] = int(self.fps_var.get())
        data["fx_intensity"] = int(self.slider_fx_intensity.get())
        data["overlay_speed"] = round(float(self.slider_ov_speed.get()), 2)
        data["overlay_opacity"] = int(self.slider_ov_opacity.get())
        return data

    def get_manual_cuts_state(self):
        """Cut-Sammlung als serialisierbarer Block (Projekt-Speicherung)."""
        if not self.manual_cuts:
            return None
        return {
            "cuts": [round(float(c), 4) for c in self.manual_cuts],
            "audio": self.manual_cuts_audio or "",
            "range": list(self.manual_cuts_range or (0.0, 0.0)),
            "info": dict(self.manual_cuts_info or {}),
            "use": bool(self.use_manual_cuts_var.get()),
        }

    def apply_manual_cuts_state(self, block):
        if not isinstance(block, dict):
            self.clear_manual_cuts()
            return
        cuts = block.get("cuts") or []
        try:
            cuts = [float(c) for c in cuts]
        except Exception:
            cuts = []
        if len(cuts) < 2:
            self.clear_manual_cuts()
            return
        self.manual_cuts = cuts
        self.manual_cuts_audio = block.get("audio") or None
        rng = block.get("range") or [0.0, 0.0]
        try:
            self.manual_cuts_range = (float(rng[0]), float(rng[1]))
        except Exception:
            self.manual_cuts_range = (0.0, 0.0)
        self.manual_cuts_info = block.get("info") or {}
        self.use_manual_cuts_var.set(bool(block.get("use", False)))
        self.refresh_manual_cuts_ui()

    def apply_preset_dict(self, data):
        if not isinstance(data, dict):
            return
        data = normalize_preset(data)
        for key, var in self._setting_vars().items():
            if key not in data:
                continue
            try:
                if isinstance(var, ctk.BooleanVar):
                    var.set(bool(data[key]))
                else:
                    var.set(data[key])
            except Exception as e:
                print(f"[Preset] '{key}' nicht uebernommen: {e}")
        if "fps" in data:
            self.fps_var.set(int(data["fps"]))
        if "fx_intensity" in data:
            val = max(20.0, min(200.0, float(data["fx_intensity"])))
            self.slider_fx_intensity.set(val)
            self.lbl_fx_intensity.configure(text=f"Effekt-Intensität: {int(val)}%")
        if "overlay_speed" in data:
            spd = max(0.2, min(4.0, float(data["overlay_speed"])))
            self.slider_ov_speed.set(spd)
            self.lbl_ov_speed.configure(text=f"Dreh-Speed: {spd:.1f}x")
        if "overlay_opacity" in data:
            op = max(10.0, min(100.0, float(data["overlay_opacity"])))
            self.slider_ov_opacity.set(op)
            self.lbl_ov_opacity.configure(text=f"Deckkraft: {int(op)}%")
        self._toggle_overlay_ui()
        self.refresh_resolution_warning()

    def _refresh_profile_menu(self, select=None):
        values = list(self.all_presets.keys())
        self.combo_profile.configure(values=values)
        if select is not None:
            self.profile_var.set(select)

    def save_current_as_profile(self):
        current = self.profile_var.get()
        suggestion = current if current in self.all_presets and current not in DEFAULT_PRESETS else ""
        hint = f"\n(leer lassen = '{suggestion}' überschreiben)" if suggestion else ""
        dialog = ctk.CTkInputDialog(title="Profil speichern", text="Name für das Profil:" + hint)
        name = dialog.get_input()
        if name is None:
            return
        name = name.strip() or suggestion
        if not name:
            return
        if name in self.all_presets and name != suggestion:
            if not messagebox.askyesno("Profil überschreiben?",
                                       f"Das Profil '{name}' existiert schon. Überschreiben?"):
                return
        self.all_presets[name] = self.get_current_settings_dict()
        self._save_presets_to_disk()
        self._refresh_profile_menu(select=name)
        self.set_status(f"Profil '{name}' gespeichert.", None)

    def import_profile_json(self):
        path = filedialog.askopenfilename(
            title="Profil aus JSON-Datei laden",
            filetypes=[("JSON-Dateien", "*.json"), ("Alle Dateien", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Die Datei enthaelt kein Profil.")
            if "mode" in data or "fps" in data or "effect_set" in data:
                base_name = os.path.splitext(os.path.basename(path))[0]
                self.all_presets[base_name] = normalize_preset(data)
                self._save_presets_to_disk()
                self._refresh_profile_menu(select=base_name)
                self.apply_preset_dict(data)
                self.set_status(f"Profil aus '{os.path.basename(path)}' geladen.", None)
            else:
                count = 0
                for name, values in data.items():
                    if isinstance(values, dict):
                        self.all_presets[name] = normalize_preset(values)
                        count += 1
                if not count:
                    raise ValueError("Keine Profile in der Datei gefunden.")
                self._save_presets_to_disk()
                self._refresh_profile_menu()
                self.set_status(f"{count} Profil(e) importiert.", None)
        except Exception as e:
            self.show_error("Fehler", f"Konnte Profil nicht laden:\n{e}")

    def export_profile_json(self):
        path = filedialog.asksaveasfilename(
            title="Aktuelles Profil exportieren",
            defaultextension=".json",
            filetypes=[("JSON-Dateien", "*.json")]
        )
        if path:
            try:
                current_data = self.get_current_settings_dict()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(current_data, f, indent=4, ensure_ascii=False)
                self.set_status(f"Profil nach '{os.path.basename(path)}' exportiert.", None)
            except Exception as e:
                self.show_error("Fehler", f"Export fehlgeschlagen:\n{e}")

    def delete_current_profile(self):
        cur = self.profile_var.get()
        if cur in DEFAULT_PRESETS:
            messagebox.showinfo("Hinweis", f"Das Standard-Profil '{cur}' kann nicht gelöscht werden.")
            return
        if cur not in self.all_presets:
            messagebox.showinfo("Hinweis", "Bitte zuerst ein eigenes Profil im Dropdown wählen.")
            return
        if not messagebox.askyesno("Profil löschen?", f"Profil '{cur}' wirklich löschen?"):
            return
        del self.all_presets[cur]
        self._save_presets_to_disk()
        self._refresh_profile_menu(select="Profil wählen...")
        self.set_status(f"Profil '{cur}' gelöscht.", None)

    # =====================================================================
    # Undo / Redo (Settings- & Medien-Historie)
    # =====================================================================
    def _register_history_traces(self):
        tracked = list(self._setting_vars().values()) + [self.fps_var, self.best_moment_len_var]
        for var in tracked:
            try:
                var.trace_add("write", self.note_change)
            except Exception:
                pass

    def capture_full_state(self):
        return {
            "settings": self.get_current_settings_dict(),
            "video_paths": list(self.video_paths),
            "audio_path": self.audio_path,
            "audio_duration": self.audio_duration,
            "start": float(self.slider_start.get()) if self.audio_duration else 0.0,
            "end": float(self.slider_end.get()) if self.audio_duration else 0.0,
            "overlay_img": self.overlay_img_path,
            "overlay_sec": float(self.slider_overlay_sec.get()),
            "manual_cuts": self.get_manual_cuts_state(),
        }

    def apply_full_state(self, state):
        if not isinstance(state, dict):
            return
        self._suspend_history = True
        try:
            self.apply_preset_dict(state.get("settings", {}))
            self.video_paths = list(state.get("video_paths", []))
            self._update_media_listbox()

            ov = state.get("overlay_img")
            if ov:
                self.overlay_img_path = ov
                self.lbl_overlay_name.configure(
                    text=os.path.basename(ov), text_color="#38bdf8"
                )
            else:
                self.overlay_img_path = None
                self.lbl_overlay_name.configure(text="Kein Bild ausgewählt", text_color="gray")

            audio = state.get("audio_path")
            if audio != self.audio_path:
                if audio:
                    self.load_audio_into_ui(audio)
                else:
                    self.reset_audio_ui()
            if self.audio_duration > 0:
                self.slider_start.set(state.get("start", 0.0) or 0.0)
                self.slider_end.set(state.get("end", self.audio_duration) or self.audio_duration)
                self.lbl_start.configure(text=f"Start: {self.slider_start.get():.1f}s")
                self.lbl_end.configure(text=f"Ende: {self.slider_end.get():.1f}s")

            sec = float(state.get("overlay_sec", 0.0))
            if hasattr(self, 'slider_overlay_sec'):
                self.slider_overlay_sec.set(sec)
                self.lbl_overlay_sec.configure(text=f"Ab Sekunde: {sec:.1f}s")

            # Cut-Sammlung zuletzt laden: sie vergleicht sich mit der gerade
            # gesetzten Musikdatei und darf sie also nicht vorher pruefen.
            if "manual_cuts" in state:
                self.apply_manual_cuts_state(state.get("manual_cuts"))

            self.refresh_resolution_warning()
            self.draw_waveform()
        finally:
            self._suspend_history = False
        self._last_state = self.capture_full_state()

    def note_change(self, *_args):
        if getattr(self, "_suspend_history", False):
            return
        if self._history_job is not None:
            return
        if hasattr(self, "_last_state"):
            self.undo_stack.append(self._last_state)
            if len(self.undo_stack) > 60:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
        self._history_job = self.after(600, self._refresh_last_state)

    def _refresh_last_state(self):
        self._history_job = None
        self._last_state = self.capture_full_state()

    def _flush_history(self):
        """
        Vor diskreten Aktionen (Datei hinzufuegen/entfernen, Cuts speichern ...)
        aufrufen: schliesst eine noch offene Slider-Aenderung ab, damit die
        Aktion ein eigener Undo-Schritt wird statt mit der vorigen zu verschmelzen.
        """
        if getattr(self, "_suspend_history", False):
            return
        if self._history_job is not None:
            try:
                self.after_cancel(self._history_job)
            except Exception:
                pass
            self._history_job = None
            self._last_state = self.capture_full_state()

    def undo_action(self):
        if not self.undo_stack:
            self.set_status("Nichts zum Rückgängigmachen.", 0.0)
            return
        self.redo_stack.append(self.capture_full_state())
        self.apply_full_state(self.undo_stack.pop())
        self.set_status(f"Rückgängig ({len(self.undo_stack)} Schritte im Verlauf).", 0.0)

    def redo_action(self):
        if not self.redo_stack:
            self.set_status("Nichts zum Wiederholen.", 0.0)
            return
        self.undo_stack.append(self.capture_full_state())
        self.apply_full_state(self.redo_stack.pop())
        self.set_status("Wiederhergestellt.", 0.0)

    # =====================================================================
    # Project Save / Load (Kompletter Arbeitsstand)
    # =====================================================================
    def save_project(self):
        current = getattr(self, "current_project_path", None)
        initial_name = os.path.basename(current) if current else (
            os.path.splitext(self._suggest_output_name())[0].replace("_edit_", "_") + ".aep"
        )
        path = filedialog.asksaveasfilename(
            title="Projekt speichern",
            initialdir=(os.path.dirname(current) if current else self._last_dir("last_project_dir")) or None,
            initialfile=initial_name,
            defaultextension=".aep",
            filetypes=[("AutoEdit Projekt", "*.aep *.json"), ("JSON-Dateien", "*.json")]
        )
        if not path:
            return
        self._remember_dir("last_project_dir", path)
        self.current_project_path = path
        try:
            base = os.path.dirname(os.path.abspath(path))

            def _rel(p):
                if not p:
                    return None
                try:
                    return os.path.relpath(os.path.abspath(p), base)
                except Exception:
                    return p

            data = {
                "format": "autoedit_project",
                "version": 3,
                "settings": self.get_current_settings_dict(),
                "videos": [_rel(p) for p in self.video_paths],
                "audio": _rel(self.audio_path),
                "overlay_img": _rel(self.overlay_img_path),
                "start": float(self.slider_start.get()) if self.audio_duration else 0.0,
                "end": float(self.slider_end.get()) if self.audio_duration else 0.0,
                "overlay_sec": float(self.slider_overlay_sec.get()),
                "manual_cuts": self.get_manual_cuts_state(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.set_status(f"Projekt gespeichert: {os.path.basename(path)}", 0.0)
        except Exception as e:
            self.show_error("Fehler", f"Projekt konnte nicht gespeichert werden:\n{e}")

    def load_project(self, path=None):
        if not path:
            path = filedialog.askopenfilename(
                title="Projekt laden",
                initialdir=self._last_dir("last_project_dir") or None,
                filetypes=[("AutoEdit Projekt", "*.aep *.json"), ("JSON-Dateien", "*.json")]
            )
        if not path:
            return
        self._remember_dir("last_project_dir", path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "settings" not in data:
                self.show_error("Fehler", "Das ist keine gültige Projektdatei.")
                return

            base = os.path.dirname(os.path.abspath(path))

            def _abs(p):
                if not p:
                    return None
                return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

            videos = [_abs(p) for p in data.get("videos", []) if _abs(p)]
            missing = [v for v in videos if not os.path.exists(v)]
            videos = [v for v in videos if os.path.exists(v)]

            cuts_block = data.get("manual_cuts")
            if isinstance(cuts_block, dict):
                audio_ref = cuts_block.get("audio") or ""
                cuts_block["audio"] = _abs(audio_ref) or audio_ref

            # Laden ist per Strg+Z rueckgaengig zu machen.
            self.undo_stack.append(self.capture_full_state())
            self.redo_stack.clear()
            self.current_project_path = path
            self.apply_full_state({
                "settings": data.get("settings", {}),
                "video_paths": videos,
                "audio_path": _abs(data.get("audio")),
                "overlay_img": _abs(data.get("overlay_img")),
                "overlay_sec": data.get("overlay_sec", 0.0),
                "start": data.get("start", 0.0),
                "end": data.get("end", 0.0),
                "manual_cuts": cuts_block,
            })

            if data.get("audio") and not os.path.exists(_abs(data.get("audio"))):
                self.show_error("Hinweis", "Die Audiodatei des Projekts wurde nicht gefunden.")
            elif missing:
                self.show_info("Projekt geladen", f"{len(missing)} Mediendatei(en) fehlten und wurden übersprungen.")
            else:
                self.set_status(f"Projekt geladen: {os.path.basename(path)}", 0.0)
        except Exception as e:
            self.show_error("Fehler", f"Projekt konnte nicht geladen werden:\n{e}")

    # =====================================================================
    # Thread-Safe UI Update Helpers
    # =====================================================================
    def _ui_after(self, delay_ms, func):
        """
        Thread-sichere UI-Aktualisierung.

        Tkinter ist nicht thread-sicher: self.after() aus einem Worker-Thread
        heraus wird vom Interpreter oft einfach ignoriert (der Callback feuert
        dann nie). Deshalb landen Aufrufe aus Fremd-Threads in einer Queue,
        die der Tk-Mainloop regelmaessig abarbeitet.
        """
        if threading.current_thread() is threading.main_thread():
            try:
                return self.after(delay_ms, func)
            except Exception:
                pass
        else:
            self._ui_queue.put((delay_ms, func))
            return None
        try:
            func()
        except Exception:
            pass
        return None

    def _drain_ui_queue(self):
        """Laeuft im Tk-Mainloop und arbeitet alle wartenden UI-Calls ab."""
        try:
            while True:
                delay, func = self._ui_queue.get_nowait()
                try:
                    func()
                except Exception as e:
                    print(f"[UI-Queue] {e}")
        except queue.Empty:
            pass
        try:
            self.after(40, self._drain_ui_queue)
        except Exception:
            pass

    def set_eta(self, text):
        self._ui_after(0, lambda: self.lbl_eta.configure(text=text))

    def set_status(self, text, progress=None):
        def _update():
            self.lbl_status.configure(text=text)
            if progress is not None:
                p_clamped = max(0.0, min(1.0, progress))
                self.progress_bar.set(p_clamped)
                self.lbl_pct.configure(text=f"{int(p_clamped * 100)}%")
        self._ui_after(0, _update)

    def show_error(self, title, msg):
        self._ui_after(0, lambda: messagebox.showerror(title, msg))

    def show_info(self, title, msg):
        self._ui_after(0, lambda: messagebox.showinfo(title, msg))

    def _center_window(self, win, min_w=0, min_h=0, max_w=1200):
        """Fenster mittig auf dem Bildschirm platzieren (auch vor dem Layout)."""
        try:
            win.update_idletasks()
            # winfo_* liefert echte Pixel, CTk-geometry() erwartet aber
            # logische Groessen und multipliziert selbst mit der Skalierung.
            # Ohne Umrechnung waren Dialoge bei 150 % Windows-Skalierung
            # anderthalbmal zu gross und lagen nicht mittig.
            scale = self._window_scaling(win)
            w = max(min_w, win.winfo_reqwidth() / scale, win.winfo_width() / scale)
            h = max(min_h, win.winfo_reqheight() / scale, win.winfo_height() / scale)
            if max_w and w > max_w:
                w = max_w
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
            w = min(w, sw / scale * 0.96)
            h = min(h, sh / scale * 0.92)
            x = max(0, int((sw - w * scale) / 2))
            y = max(0, int((sh - h * scale) / 2))
            win.geometry(f"{int(w)}x{int(h)}+{x}+{y}")
        except Exception as e:
            print(f"[UI] Zentrieren fehlgeschlagen: {e}")

    def show_render_done(self, out_path, extra_msg=""):
        """
        Fertig-Dialog mit Direktzugriff: ansehen, Ordner oeffnen, schliessen.
        Kein modaler Messagebox-Block, der den ganzen Workflow stoppt.
        """
        def _build():
            if self._done_win is not None and self._done_win.winfo_exists():
                try:
                    self._done_win.destroy()
                except Exception:
                    pass

            win = ctk.CTkToplevel(self)
            self._done_win = win
            win.title("Fertig")
            win.resizable(False, False)
            win.transient(self)
            win.attributes("-topmost", True)

            ctk.CTkLabel(
                win,
                text="Video erfolgreich gespeichert",
                font=("Arial", 16, "bold"),
                text_color="#22c55e"
            ).pack(pady=(18, 6))

            name_lbl = ctk.CTkLabel(win, text=os.path.basename(out_path), font=("Arial", 12, "bold"))
            name_lbl.pack(pady=(0, 2))

            path_lbl = ctk.CTkLabel(
                win, text=os.path.dirname(out_path),
                text_color="#94a3b8", font=("Arial", 10), wraplength=520, justify="center"
            )
            path_lbl.pack(pady=(0, 4))

            if extra_msg:
                ctk.CTkLabel(win, text=extra_msg, text_color="#38bdf8", font=("Arial", 11)).pack(pady=(0, 6))

            try:
                dur_txt = ""
                probe = VideoFileClip(out_path)
                dur_txt = f"Länge: {probe.duration:.1f}s | {probe.size[0]}x{probe.size[1]}"
                probe.close()
                ctk.CTkLabel(win, text=dur_txt, text_color="#94a3b8", font=("Arial", 11)).pack(pady=(0, 8))
            except Exception:
                pass

            btn_row = ctk.CTkFrame(win, fg_color="transparent")
            btn_row.pack(pady=(10, 4))

            def _play():
                win.destroy()
                self._done_win = None
                try:
                    os.startfile(out_path)
                except Exception as e:
                    self.show_error("Fehler", f"Video konnte nicht geoeffnet werden:\n{e}")

            def _open_folder():
                try:
                    subprocess.Popen(f'explorer /select,"{os.path.normpath(out_path)}"')
                except Exception:
                    os.startfile(os.path.dirname(out_path))

            def _close():
                win.destroy()
                self._done_win = None

            ctk.CTkButton(
                btn_row, text="Direkt ansehen", command=_play,
                fg_color="#2563eb", hover_color="#1d4ed8", width=170, height=40,
                font=("Arial", 14, "bold")
            ).pack(side="left", padx=8)

            ctk.CTkButton(
                btn_row, text="Öffnen", command=_open_folder,
                fg_color="#2b719e", hover_color="#1f5375", width=150, height=40
            ).pack(side="left", padx=8)

            ctk.CTkButton(
                btn_row, text="Schliessen", command=_close,
                fg_color="#444444", hover_color="#555555", width=120, height=40
            ).pack(side="left", padx=8)

            win.protocol("WM_DELETE_WINDOW", _close)
            self._center_window(win, min_w=520, min_h=260)
            win.after(250, lambda: win.attributes("-topmost", False))
            return win

        self._ui_after(0, _build)

    # =====================================================================
    # Media Selection & Preview
    # =====================================================================
    def select_videos(self):
        vid = " ".join(f"*{e}" for e in VIDEO_EXTS)
        img = " ".join(f"*{e}" for e in IMAGE_EXTS)
        paths = filedialog.askopenfilenames(
            title="Wähle Videos oder Bilder aus",
            initialdir=self._last_dir("last_media_dir") or None,
            filetypes=[
                ("Medien-Dateien", f"{vid} {img}"),
                ("Video-Dateien", vid),
                ("Bilder", img),
                ("Alle Dateien", "*.*")
            ]
        )
        if paths:
            self._remember_dir("last_media_dir", paths[0])
            self.add_media_paths(list(paths))

    def clear_videos(self):
        if not self.video_paths:
            return
        self._flush_history()
        self.video_paths = []
        self._update_media_listbox()
        self.refresh_resolution_warning()
        self.note_change()
        self.set_status("Medienliste geleert (Strg+Z holt sie zurueck).", None)

    def start_manual_recording(self, render_after=None):
        """
        Startet die Aufnahme. Die Cuts landen in der Sammlung; kam der Aufruf
        von 'Video Rendern', wird danach direkt in render_after gerendert.
        """
        if not self.audio_ok:
            self.show_error("Hinweis", "Kein Audiogeraet verfuegbar - die Live-Aufnahme braucht Tonausgabe.")
            return
        if not self.audio_path:
            self.show_error("Hinweis", "Bitte zuerst eine Audio-Datei auswählen.")
            return
        if not self.video_paths:
            self.show_error("Hinweis", "Bitte zuerst mindestens ein Video oder Bild wählen.")
            return
        start_t = float(self.slider_start.get())
        end_t = float(self.slider_end.get())
        if end_t - start_t < 1.0:
            self.show_error("Fehler", "Der Musik-Bereich muss mindestens eine Sekunde lang sein.")
            return

        self.is_cancelled = False
        self._set_render_buttons(True)
        self.open_manual_recorder(start_t, end_t, render_after=render_after)

    def _prepare_manual_audio(self, start_t, end_t, on_ready):
        """
        Schneidet die Passage als WAV aus und sucht Onsets - im Hintergrund,
        weil HPSS bei laengeren Passagen mehrere Sekunden dauert und das
        Fenster sonst einfriert. on_ready(temp_dir, wav, onsets) laeuft im
        UI-Thread; bei Fehlern werden die Buttons wieder freigegeben.
        """
        self.stop_audio_preview()
        audio_source = self.audio_path
        total_duration = end_t - start_t
        self.set_status("Bereite Audio-Passage vor (Onset-Analyse)...", 0.05)

        def _worker():
            temp_dir = tempfile.mkdtemp(prefix="autoedit_manual_")
            wav = os.path.join(temp_dir, "master_audio.wav")
            try:
                cmd = [
                    self.ffmpeg_exe, "-y",
                    "-ss", f"{start_t:.3f}",
                    "-i", audio_source,
                    "-t", f"{total_duration:.3f}",
                    "-vn", "-ac", "2", "-ar", "44100",
                    "-acodec", "pcm_s16le",
                    wav
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=_NO_WINDOW)
                if res.returncode != 0 or not os.path.exists(wav):
                    raise RuntimeError("FFmpeg Audio-Extraktion fehlgeschlagen:\n" + _ffmpeg_tail(res.stderr))
            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                err = str(e)

                def _fail():
                    self._set_render_buttons(False)
                    self.set_status("Audio-Vorbereitung fehlgeschlagen.", 0.0)
                    self.show_error("Fehler", f"Audio-Vorbereitung fehlgeschlagen:\n{err}")

                self._ui_after(0, _fail)
                return

            onsets = []
            try:
                y_m, sr_m = librosa.load(wav, sr=22050)
                y_perc = librosa.effects.percussive(y_m, margin=2.0)
                env_low = librosa.onset.onset_strength(y=y_perc, sr=sr_m, fmax=280)
                env_high = librosa.onset.onset_strength(y=y_perc, sr=sr_m, fmin=800, fmax=5500)
                onset_env = env_low * 1.0 + env_high * 0.65
                onset_frames = librosa.onset.onset_detect(
                    onset_envelope=onset_env, sr=sr_m, hop_length=512,
                    backtrack=True, wait=6, delta=0.07
                )
                onsets = [float(t) for t in librosa.frames_to_time(onset_frames, sr=sr_m, hop_length=512)]
            except Exception as e:
                print(f"[Onsets] {e}")
                onsets = []

            self._ui_after(0, lambda: (self.set_status("Bereit.", 0.0), on_ready(temp_dir, wav, onsets)))

        threading.Thread(target=_worker, daemon=True).start()

    def open_saved_cuts_review(self):
        """Review-Fenster fuer die bereits gespeicherten Cuts erneut oeffnen."""
        if len(self.manual_cuts) < 2:
            self.show_info("Hinweis", "Es sind noch keine Cuts gespeichert.\n\n"
                                      "Nutze 'Jetzt aufnehmen' in Punkt 3.")
            return
        if not self.audio_ok:
            self.show_error("Hinweis", "Kein Audiogeraet verfuegbar.")
            return
        if not self.audio_path:
            self.show_error("Hinweis", "Bitte zuerst die passende Musikdatei wählen.")
            return
        if self.manual_cuts_audio and os.path.normcase(os.path.abspath(self.manual_cuts_audio)) != \
                os.path.normcase(os.path.abspath(self.audio_path)):
            self.show_info(
                "Hinweis",
                "Die gespeicherten Cuts gehoeren zu:\n"
                f"{os.path.basename(self.manual_cuts_audio)}\n\n"
                "Aktuell geladen ist eine andere Musik. Bitte diese zuerst waehlen."
            )
            return

        rng = self.manual_cuts_range or (0.0, self.audio_duration)
        start_t, end_t = float(rng[0]), float(rng[1])
        if end_t <= start_t:
            start_t, end_t = 0.0, float(self.audio_duration)

        # Wichtig: exakt die aufgezeichnete Passage ausschneiden - die Marker
        # sitzen relativ zum Aufnahmefenster.
        self.manual_review_source = self.audio_path
        self._set_render_buttons(True)

        def _ready(temp_dir, wav, onsets):
            self.open_manual_review(
                start_t, end_t,
                list(self.manual_cuts), list(self.manual_cuts),
                self.manual_cuts_info.get("snapped", 0),
                self.manual_cuts_info.get("loose", 0),
                wav,
                onsets=onsets
            )

        self._prepare_manual_audio(start_t, end_t, _ready)

    def source_max_resolution(self):
        best = (0, 0)
        for path in list(self.video_paths)[:40]:
            w, h, _ = self.media_info(path)
            if w * h > best[0] * best[1]:
                best = (int(w), int(h))
        return best

    def refresh_resolution_warning(self):
        lbl = getattr(self, "lbl_res_warn", None)
        if lbl is None:
            return
        choice = self.res_var.get()
        if "Auto" in choice:
            lbl.configure(text="")
            return
        src_w, src_h = self.source_max_resolution()
        if src_w <= 0:
            lbl.configure(text="")
            return
        # Hochformat dreht die Achsen - verglichen wird nach der Pixelzahl.
        src_px = src_w * src_h
        if "1440p" in choice:
            want_px = 2560 * 1440
        elif "1080p" in choice:
            want_px = 1920 * 1080
        else:
            lbl.configure(text="")
            return
        if want_px > src_px * 1.02:
            lbl.configure(
                text=f"Achtung: Quellen sind nur {src_w}x{src_h} - "
                     f"{'1440p' if '1440p' in choice else '1080p'} erzwingt Hochskalierung "
                     f"(sieht weicher aus)."
            )
        else:
            lbl.configure(text="")

    def media_info(self, path):
        """(breite, hoehe, dauer) - gecacht, per OpenCV statt ffmpeg-Prozess."""
        if path in self._media_info_cache:
            return self._media_info_cache[path]
        info = (0, 0, 0.0)
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in IMAGE_EXTS:
                with PIL.Image.open(path) as im:
                    info = (int(im.size[0]), int(im.size[1]), 0.0)
            else:
                cap = cv2.VideoCapture(path)
                try:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                    dur = frames / fps if fps > 0 else 0.0
                finally:
                    cap.release()
                if w <= 0 or h <= 0:
                    probe = VideoFileClip(path)
                    w, h = probe.size
                    dur = float(probe.duration or 0.0)
                    probe.close()
                info = (int(w), int(h), float(dur))
        except Exception:
            info = (0, 0, 0.0)
        self._media_info_cache[path] = info
        return info

    def add_media_paths(self, paths):
        self._flush_history()
        known = {os.path.normcase(os.path.abspath(p)) for p in self.video_paths}
        added = 0
        skipped = 0
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext not in VIDEO_EXTS + IMAGE_EXTS:
                skipped += 1
                continue
            key = os.path.normcase(os.path.abspath(p))
            if key in known or not os.path.exists(p):
                continue
            known.add(key)
            self.video_paths.append(p)
            added += 1
        if added:
            self._update_media_listbox()
            self.refresh_resolution_warning()
            self.note_change()
            msg = f"{added} Datei(en) hinzugefuegt."
            if skipped:
                msg += f" {skipped} nicht unterstuetzte uebersprungen."
            self.set_status(msg, None)
        elif skipped:
            self.set_status(f"{skipped} Datei(en) nicht unterstuetzt (Videos: MP4/MOV/MKV/AVI/WEBM, Bilder: PNG/JPG/WEBP/BMP).", None)
        return added

    def remove_media_path(self, path):
        self._flush_history()
        if path in self.video_paths:
            self.video_paths.remove(path)
            self._update_media_listbox()
            self.refresh_resolution_warning()
            self.note_change()

    def _update_media_listbox(self):
        for row in getattr(self, "_media_rows", []):
            try:
                row.destroy()
            except Exception:
                pass
        self._media_rows = []
        count = len(self.video_paths)
        if count == 0:
            hint = ctk.CTkLabel(
                self.media_list_frame,
                text="Noch keine Dateien.\n'Clips / Bilder wählen'"
                     + (" oder Dateien hierher ziehen." if self.dnd_enabled else "")
                     + "\nMP4, MOV, MKV, AVI, WEBM, PNG, JPG, WEBP",
                text_color="#64748b", font=("Arial", 11), justify="left", anchor="w"
            )
            hint.pack(fill="x", padx=6, pady=6)
            self._media_rows.append(hint)
            self.lbl_videos_count.configure(text="0 Dateien ausgewählt", text_color="gray")
            return

        n_img = 0
        for i, p in enumerate(self.video_paths):
            is_img = os.path.splitext(p)[1].lower() in IMAGE_EXTS
            n_img += int(is_img)
            row = ctk.CTkFrame(self.media_list_frame, fg_color="transparent", height=24)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(
                row, text=f"{i + 1:>2}", width=24, anchor="e",
                font=("Consolas", 11), text_color="#475569"
            ).pack(side="left")
            ctk.CTkLabel(
                row, text="BILD" if is_img else "VIDEO", width=46,
                font=("Arial", 9, "bold"), text_color="#a855f7" if is_img else "#38bdf8"
            ).pack(side="left", padx=(4, 4))
            ctk.CTkLabel(
                row, text=os.path.basename(p), anchor="w", font=("Arial", 11)
            ).pack(side="left", fill="x", expand=True)
            ctk.CTkButton(
                row, text="x", width=26, height=22, fg_color="#334155", hover_color="#8b1e1e",
                command=lambda path=p: self.remove_media_path(path)
            ).pack(side="right", padx=(4, 2))
            self._media_rows.append(row)

        n_vid = count - n_img
        parts = []
        if n_vid:
            parts.append(f"{n_vid} Video{'s' if n_vid != 1 else ''}")
        if n_img:
            parts.append(f"{n_img} Bild{'er' if n_img != 1 else ''}")
        self.lbl_videos_count.configure(text=" + ".join(parts) + " bereit", text_color="#38bdf8")

    def select_audio(self):
        exts = " ".join(f"*{e}" for e in AUDIO_EXTS)
        path = filedialog.askopenfilename(
            title="Wähle eine Musik-Datei",
            initialdir=self._last_dir("last_audio_dir") or None,
            filetypes=[("Audio-Dateien", exts), ("Alle Dateien", "*.*")]
        )
        if path:
            self._remember_dir("last_audio_dir", path)
            self._flush_history()
            if self.load_audio_into_ui(path):
                self.note_change()

    def reset_audio_ui(self):
        """Zustand ohne Song (z. B. nach Strg+Z ueber das Laden hinaus)."""
        self.stop_audio_preview()
        self.audio_path = None
        self.audio_duration = 0.0
        self.wave_env = None
        self.wave_beats = []
        self.wave_drop = 0.0
        self.drop_candidates = []
        self.wave_view = (0.0, 1.0)
        self.slider_start.configure(to=100)
        self.slider_end.configure(to=100)
        self.slider_start.set(0)
        self.slider_end.set(0)
        self.lbl_start.configure(text="Start: 0.0s")
        self.lbl_end.configure(text="Ende: 0.0s")
        self.lbl_audio_status.configure(text="Kein Song gewählt", text_color="gray")
        self.btn_audio_preview.configure(state="disabled")
        self.refresh_manual_cuts_ui()
        self.draw_waveform()

    def load_audio_into_ui(self, path):
        if not path or not os.path.exists(path):
            if path:
                self.show_error("Fehler", f"Audiodatei nicht gefunden:\n{path}")
            return False
        self.stop_audio_preview()
        duration = 0.0
        try:
            duration = float(librosa.get_duration(path=path))
        except Exception:
            try:
                probe = AudioFileClip(path)
                duration = float(probe.duration)
                probe.close()
            except Exception as e:
                self.show_error("Fehler", f"Konnte die Audiodatei nicht lesen:\n{os.path.basename(path)}\n\n{e}")
                return False
        if duration < 1.0:
            self.show_error("Fehler", "Die Audiodatei ist kürzer als eine Sekunde.")
            return False

        self.audio_path = path
        self.audio_duration = duration
        self.slider_start.configure(to=self.audio_duration)
        self.slider_end.configure(to=self.audio_duration)
        self.slider_start.set(0)
        self.slider_end.set(self.audio_duration)
        self.wave_view = (0.0, self.audio_duration)
        self.lbl_start.configure(text="Start: 0.0s")
        self.lbl_end.configure(text=f"Ende: {self.audio_duration:.1f}s")
        if hasattr(self, 'slider_overlay_sec'):
            self.slider_overlay_sec.configure(to=max(1.0, self.audio_duration))
        mins, secs = divmod(int(round(self.audio_duration)), 60)
        self.lbl_audio_status.configure(
            text=f"{os.path.basename(path)} ({mins}:{secs:02d})",
            text_color="#38bdf8"
        )
        # Musikwechsel: gespeicherte Cuts koennen zu einer anderen Datei gehoeren.
        self.refresh_manual_cuts_ui()
        self.btn_audio_preview.configure(state="normal" if self.audio_ok else "disabled")

        self.wave_env = None
        self.wave_beats = []
        self.wave_drop = 0.0
        self.drop_candidates = []
        self.drop_scan_offset = 0.0
        self.draw_waveform()
        self.analyze_waveform_async()
        self.set_status(f"Song geladen: {os.path.basename(path)} - Wellenform wird analysiert...", None)
        return True

    def _range_changed(self):
        """Gemeinsamer Nachlauf, wenn Start oder Ende sich aendert."""
        self.lbl_start.configure(text=f"Start: {self.slider_start.get():.1f}s")
        self.lbl_end.configure(text=f"Ende: {self.slider_end.get():.1f}s")
        self.draw_waveform()
        self.refresh_manual_cuts_ui()
        self.note_change()

    def set_range(self, start=None, end=None):
        if not self.audio_duration:
            return
        s = float(self.slider_start.get()) if start is None else float(start)
        e = float(self.slider_end.get()) if end is None else float(end)
        s = max(0.0, min(self.audio_duration, s))
        e = max(0.0, min(self.audio_duration, e))
        if e - s < 0.5:
            if end is not None and start is None:
                s = max(0.0, e - 0.5)
            else:
                e = min(self.audio_duration, s + 0.5)
                s = min(s, max(0.0, e - 0.5))
        self.slider_start.set(s)
        self.slider_end.set(e)
        self._range_changed()

    def update_start_lbl(self, val):
        self.set_range(start=float(val))

    def update_end_lbl(self, val):
        self.set_range(end=float(val))

    # =====================================================================
    # Waveform View (Beat- & Drop-Marker)
    # =====================================================================
    def analyze_waveform_async(self):
        """
        Zweistufig: erst die Huelle (schnell, sofort sichtbar), dann Beats und
        Drop. Jeder Lauf gehoert zu genau einer Datei - wird waehrenddessen
        ein anderer Song geladen, verwirft der alte Lauf sein Ergebnis, statt
        (wie frueher) den neuen Lauf zu blockieren.
        """
        if not self.audio_path:
            return
        path = self.audio_path
        self.wave_analysis_running = True
        self.wave_status = "Wellenform wird geladen..."

        def _hint(drop):
            base = "Links: Start | Rechts: Ende | Mausrad: Zoom | Mitte: Verschieben"
            if drop > 0:
                self.lbl_waveform_hint.configure(
                    text=f"Links: Start | Rechts: Ende | Mausrad: Zoom | KI-Drop bei {drop:.1f}s",
                    text_color="#f97316"
                )
            else:
                self.lbl_waveform_hint.configure(text=base, text_color="#94a3b8")

        def _worker():
            sr = 22050
            try:
                y, _ = librosa.load(path, sr=sr, duration=1200.0, mono=True)
            except Exception as e:
                print(f"[Waveform Error] {e}")
                y = None

            env, hop = None, 0.0
            if y is not None and len(y) >= sr // 2:
                block = max(256, int(sr * 0.02))
                n_blocks = len(y) // block
                if n_blocks >= 4:
                    env = np.abs(y[:n_blocks * block].reshape(n_blocks, block)).max(axis=1)
                    hop = block / float(sr)

            def _apply_env():
                if path != self.audio_path:
                    return
                self.wave_env = env
                self.wave_hop = hop
                self.wave_status = "Wellenform nicht verfuegbar." if env is None else ""
                self.draw_waveform()

            self._ui_after(0, _apply_env)
            if env is None:
                self._ui_after(0, lambda: setattr(self, "wave_analysis_running", False))
                return

            beats, drop = [], 0.0
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
                _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=512)
                beats = [float(b) for b in librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)]
            except Exception as e:
                print(f"[Waveform Beats] {e}")
            try:
                drop = float(ai_detect_beat_drop(y, sr, len(y) / float(sr)))
            except Exception as e:
                print(f"[Waveform Drop] {e}")
            del y

            def _apply_beats():
                if path != self.audio_path:
                    return
                self.wave_analysis_running = False
                self.wave_beats = beats
                self.wave_drop = drop
                self.draw_waveform()
                _hint(drop)
                self.set_status(
                    f"Analyse fertig: {len(beats)} Beats"
                    + (f", KI-Drop bei {drop:.1f}s" if drop > 0 else "")
                    + ". Bereich in der Wellenform waehlen.", None
                )

            self._ui_after(0, _apply_beats)

        threading.Thread(target=_worker, daemon=True).start()

    def _wave_view(self):
        """Sichtbarer Zeitbereich der Wellenform (Zoom)."""
        dur = self.audio_duration
        v0, v1 = getattr(self, "wave_view", (0.0, dur))
        if not dur:
            return 0.0, 1.0
        span = max(0.05, min(dur, v1 - v0))
        v0 = max(0.0, min(dur - span, v0))
        return v0, v0 + span

    def _wave_set_view(self, v0, v1):
        dur = self.audio_duration
        if not dur:
            return
        span = max(0.05, min(dur, v1 - v0))
        v0 = max(0.0, min(dur - span, v0))
        self.wave_view = (v0, v0 + span)
        self.draw_waveform()

    def on_waveform_wheel(self, event):
        if not self.audio_duration:
            return
        v0, v1 = self._wave_view()
        t = self._wave_time_from_x(event)
        if t is None:
            return
        span = v1 - v0
        factor = 0.8 if event.delta > 0 else 1.25
        new_span = max(0.08, min(self.audio_duration, span * factor))
        ratio = (t - v0) / span if span > 0 else 0.5
        new_v0 = t - ratio * new_span
        self._wave_set_view(new_v0, new_v0 + new_span)

    def on_waveform_pan(self, event):
        if not self.audio_duration:
            return
        v0, v1 = self._wave_view()
        span = v1 - v0
        w = self._wave_width()
        delta_t = -(event.x - self._pan_last_x) / float(w) * span if w else 0
        self._pan_last_x = event.x
        self._wave_set_view(v0 + delta_t, v0 + delta_t + span)

    def on_waveform_pan_start(self, event):
        self._pan_last_x = event.x

    def on_waveform_reset_view(self, event=None):
        self.wave_view = (0.0, self.audio_duration)
        self.draw_waveform()

    def _wave_width(self):
        """Canvas-Breite; winfo_width() ist vor dem Layout 1, dann waere die
        Wellenform unsichtbar. Auf die angeforderte Breite zurueckfallen."""
        canvas = self.wave_canvas
        w = canvas.winfo_width()
        if w <= 2:
            w = canvas.winfo_reqwidth()
        return max(300, w)

    def draw_waveform(self):
        if not hasattr(self, "wave_canvas"):
            return
        canvas = self.wave_canvas
        canvas.delete("all")
        w = self._wave_width()
        h = canvas.winfo_height()
        if h <= 2:
            h = canvas.winfo_reqheight() or 70
        if w <= 2 or h <= 2:
            return

        dur = self.audio_duration
        if not dur or self.wave_env is None or len(self.wave_env) == 0:
            if not dur:
                msg = "Song wählen" + (" oder hierher ziehen" if self.dnd_enabled else "") + " - dann Bereich hier markieren"
            else:
                msg = getattr(self, "wave_status", "") or "Wellenform wird analysiert..."
            canvas.create_text(
                w // 2, h // 2, text=msg, fill="#475569", font=("Arial", 10)
            )
            return

        env = self.wave_env
        hop = max(1e-4, self.wave_hop)
        mid = h / 2.0
        max_v = float(np.percentile(env, 98)) or 1.0
        start_offset = self.drop_scan_offset or 0.0

        v0, v1 = self._wave_view()
        span = max(1e-6, v1 - v0)

        def x_of(t):
            return int((t - v0) / span * w)

        for b in self.wave_beats:
            x = x_of(b)
            if 0 <= x < w:
                canvas.create_line(x, 0, x, h, fill="#16243b")

        top_pts = []
        bot_pts = []
        i_start = max(0, int(v0 / hop))
        i_end = min(len(env), int(v1 / hop) + 2)
        for x in range(0, w, 1):
            t_a = v0 + (x / float(w)) * span
            t_b = v0 + ((x + 1) / float(w)) * span
            i0 = max(0, min(len(env) - 1, int(t_a / hop)))
            i1 = max(i0 + 1, min(len(env), int(t_b / hop) + 1))
            seg = env[i0:i1]
            v = float(np.max(seg)) / max_v if len(seg) else 0.0
            amp = min(1.0, v) * (mid - 2)
            top_pts.extend((x, mid - amp))
            bot_pts.extend((x, mid + amp))

        if top_pts:
            polygon = top_pts + list(reversed(bot_pts))
            canvas.create_polygon(polygon, fill="#38bdf8", outline="")

        xs = x_of(self.slider_start.get())
        xe = x_of(self.slider_end.get())
        if xs > 0:
            canvas.create_rectangle(0, 0, xs, h, fill="#020617", stipple="gray50", outline="")
        if xe < w:
            canvas.create_rectangle(xe, 0, w, h, fill="#020617", stipple="gray50", outline="")

        canvas.create_line(xs, 0, xs, h, fill="#22c55e", width=2)
        canvas.create_line(xe, 0, xe, h, fill="#f97316", width=2)

        # Zoom-Anzeige
        canvas.create_text(
            6, h - 9, text=f"{v0:.2f}s - {v1:.2f}s",
            fill="#475569", font=("Consolas", 9), anchor="w"
        )

        if self.wave_drop > 0:
            xd = x_of(self.wave_drop)
            if 0 <= xd < w:
                canvas.create_line(xd, 0, xd, h, fill="#ef4444", width=2)
                label_x = xd + 4 if xd < w - 46 else xd - 4
                canvas.create_text(
                    label_x, 8, text="DROP", fill="#ef4444",
                    font=("Arial", 8, "bold"), anchor="w" if label_x > xd else "e"
                )

        for cand in self.drop_candidates:
            t = start_offset + cand["time"]
            x = x_of(t)
            if not (0 <= x < w):
                continue
            if cand.get("is_best"):
                canvas.create_line(x, 0, x, h, fill="#f0abfc", width=3)
                canvas.create_oval(x - 5, 2, x + 5, 12, fill="#f0abfc", outline="")
                canvas.create_text(
                    x + 6 if x < w - 60 else x - 6, h - 9, text="BEST",
                    fill="#f0abfc", font=("Arial", 8, "bold"),
                    anchor="w" if x < w - 60 else "e"
                )
            else:
                canvas.create_line(x, 0, x, h, fill="#a855f7", width=1, dash=(3, 3))
                canvas.create_oval(x - 3, 3, x + 3, 9, fill="#a855f7", outline="")

        if self.wave_playhead is not None:
            self._draw_playhead(self.wave_playhead)

    def _draw_playhead(self, t_abs):
        """Abspiel-Cursor in der Wellenform (eigenes Canvas-Item, kein Redraw)."""
        self.wave_playhead = t_abs
        if not hasattr(self, "wave_canvas"):
            return
        canvas = self.wave_canvas
        canvas.delete("playhead")
        self.wave_playhead = None
        if t_abs is None or not self.audio_duration:
            return
        w = self._wave_width()
        h = canvas.winfo_height()
        if h <= 2:
            h = canvas.winfo_reqheight() or 70
        if w <= 2 or h <= 2:
            return
        v0, v1 = self._wave_view()
        x = int((t_abs - v0) / max(1e-6, v1 - v0) * w)
        if not (0 <= x < w):
            return
        canvas.create_line(x, 0, x, h, fill="#facc15", width=2, tags="playhead")
        canvas.create_oval(
            x - 4, 0, x + 4, 8, fill="#facc15", outline="", tags="playhead"
        )
        self.wave_playhead = t_abs

    def clear_playhead(self):
        self.wave_playhead = None
        if hasattr(self, "wave_canvas"):
            self.wave_canvas.delete("playhead")

    def _wave_time_from_x(self, event):
        if not self.audio_duration:
            return None
        v0, v1 = self._wave_view()
        frac = max(0.0, min(1.0, event.x / float(self._wave_width())))
        return max(0.0, min(self.audio_duration, v0 + frac * (v1 - v0)))

    def on_waveform_click(self, event):
        t = self._wave_time_from_x(event)
        if t is None:
            return
        self.set_range(start=t)
        if self.is_preview_playing:
            self.start_audio_preview()

    def on_waveform_right_click(self, event):
        t = self._wave_time_from_x(event)
        if t is None:
            return
        self.set_range(end=t)

    # =====================================================================
    # Drop-Kandidaten-Analyse (zweite KI-Stufe)
    # =====================================================================
    def scan_drop_candidates_ui(self):
        if not self.audio_path or not os.path.exists(self.audio_path):
            self.show_error("Hinweis", "Bitte zuerst eine Audio-Datei auswählen.")
            return

        start_t = float(self.slider_start.get())
        end_t = float(self.slider_end.get())
        scan_len = max(1.0, end_t - start_t)
        if scan_len < 4.0:
            self.show_error("Hinweis", "Der Auswahlbereich ist zu kurz (mindestens 4 Sekunden).")
            return

        self.btn_candidates.configure(state="disabled", text="Analysiere...")
        self.set_status("KI durchsucht den Auswahlbereich nach Drop-Kandidaten...", 0.25)

        def _worker():
            t0 = time.time()
            try:
                y, sr = librosa.load(self.audio_path, offset=start_t, duration=scan_len, sr=22050)
                found = ai_find_drop_candidates(y, sr, len(y) / float(sr), max_candidates=8)
            except Exception as e:
                print(f"[Candidate Scan Error] {e}")
                found = []

            for item in found:
                item["abs_time"] = start_t + item["time"]

            def _apply():
                self.btn_candidates.configure(state="normal", text="Drop-Kandidaten suchen")
                self.drop_candidates = found
                self.drop_scan_offset = start_t
                self.draw_waveform()

                if not found:
                    self.set_status("Keine Drop-Kandidaten im Auswahlbereich gefunden.", 0.0)
                    return

                best = found[0]
                self.wave_drop = best["abs_time"]
                self.slider_overlay_sec.set(best["time"])
                self.lbl_overlay_sec.configure(text=f"Ab Sekunde: {best['time']:.1f}s")
                self.overlay_start_mode.set("custom")
                self.set_status(
                    f"{len(found)} Drop-Kandidaten | bester bei {best['time']:.1f}s "
                    f"({time.time() - t0:.1f}s)", 0.0
                )
                self.show_drop_candidates(start_t, scan_len)

            self._ui_after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def show_drop_candidates(self, scan_start, scan_len):
        if self._candidate_win is not None:
            try:
                if self._candidate_win.winfo_exists():
                    self._candidate_win.destroy()
            except Exception:
                pass

        win = ctk.CTkToplevel(self)
        self._candidate_win = win
        win.title("Drop-Kandidaten")
        win.resizable(False, False)
        win.transient(self)

        ctk.CTkLabel(
            win, text="Drop-Kandidaten", font=("Arial", 17, "bold")
        ).pack(pady=(16, 2))

        ctk.CTkLabel(
            win,
            text=f"Auswahlbereich: {scan_start:.1f}s - {scan_start + scan_len:.1f}s",
            font=("Arial", 11), text_color="#94a3b8"
        ).pack(pady=(0, 2))

        ctk.CTkLabel(
            win,
            text="Bester Kandidat ist bereits gesetzt. Zeile anklicken zum Uebernehmen.",
            font=("Arial", 11), text_color="#38bdf8"
        ).pack(pady=(0, 10))

        rows = ctk.CTkFrame(win, fg_color="transparent")
        rows.pack(fill="x", padx=16)

        for idx, cand in enumerate(self.drop_candidates):
            conf = float(cand.get("confidence", 0.0))
            is_best = bool(cand.get("is_best"))

            row = ctk.CTkFrame(rows, fg_color="transparent")
            row.pack(fill="x", pady=2)

            tag = "BEST" if is_best else f"#{idx + 1}"
            color = "#f0abfc" if is_best else "#e2e8f0"

            btn = ctk.CTkButton(
                row,
                text=f"{tag:<5}",
                width=58, height=30,
                fg_color="#7e22ce" if is_best else "#334155",
                hover_color="#6b21a8" if is_best else "#475569",
                command=lambda c=cand: self._apply_drop_candidate(c)
            )
            btn.pack(side="left", padx=(0, 8))

            ctk.CTkLabel(
                row, text=f"{cand['time']:>6.2f}s", width=70, anchor="w",
                font=("Consolas", 13, "bold"), text_color=color
            ).pack(side="left")

            bar_w = int(120 * max(0.05, conf))
            bar = ctk.CTkProgressBar(row, width=120, height=14, progress_color="#a855f7")
            bar.pack(side="left", padx=8)
            bar.set(conf)

            ctk.CTkLabel(
                row, text=f"{conf * 100:>3.0f}%", width=40, anchor="w",
                font=("Consolas", 11), text_color="#94a3b8"
            ).pack(side="left")

            note = "Intro-Regel" if cand.get("intro_promoted") else ""
            ctk.CTkLabel(
                row, text=note, width=90, anchor="w",
                font=("Arial", 10), text_color="#38bdf8"
            ).pack(side="left", padx=6)

            ctk.CTkLabel(
                row, text=f"Sprung {cand.get('jump', 0):.2f}", anchor="e",
                font=("Arial", 10), text_color="#475569"
            ).pack(side="right", padx=6)

        ctk.CTkButton(
            win, text="Schließen", command=lambda: (win.destroy(), setattr(self, "_candidate_win", None)),
            fg_color="#444444", hover_color="#555555", height=34, width=150
        ).pack(pady=12)
        self._center_window(win, min_w=480)

    def _apply_drop_candidate(self, cand):
        t = float(cand["time"])
        self.slider_overlay_sec.set(t)
        self.lbl_overlay_sec.configure(text=f"Ab Sekunde: {t:.1f}s")
        self.overlay_start_mode.set("custom")
        self.wave_drop = float(cand.get("abs_time", self.drop_scan_offset + t))
        self.draw_waveform()
        self.set_status(f"Drop-Kandidat bei {t:.1f}s uebernommen.", 0.0)

    # =====================================================================
    # Best-Moment-Finder
    # =====================================================================
    def find_best_moment_ui(self):
        if not self.audio_path or not os.path.exists(self.audio_path):
            self.show_error("Hinweis", "Bitte zuerst eine Audio-Datei auswählen.")
            return

        target_len = float(self.best_moment_len_var.get().rstrip("s"))

        def _worker():
            t0 = time.time()
            self.set_status("KI sucht die energiereichste Passage...", 0.3)
            result = ai_find_best_moment(self.audio_path, target_len)

            def _apply():
                if not result:
                    self.set_status("Keine geeignete Passage gefunden.", 0.0)
                    return
                start, end, drop = result
                self.set_range(start=start, end=end)
                # Overlay-Sekunden sind relativ zum Edit-Start.
                if start < drop < end:
                    rel = drop - start
                    self.slider_overlay_sec.set(rel)
                    self.lbl_overlay_sec.configure(text=f"Ab Sekunde: {rel:.1f}s")
                    self.wave_drop = drop
                    self.draw_waveform()
                self.set_status(
                    f"Bester Hook: {start:.1f}s - {end:.1f}s ({time.time() - t0:.1f}s)", 0.0
                )

            self._ui_after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def toggle_audio_preview(self):
        if self.is_preview_playing:
            self.stop_audio_preview()
        else:
            self.start_audio_preview()

    def start_audio_preview(self):
        if not self.audio_path or not os.path.exists(self.audio_path):
            return

        self.stop_audio_preview()

        start_t = self.slider_start.get()
        end_t = self.slider_end.get()
        dur = max(0.5, end_t - start_t)

        try:
            fd, temp_preview = tempfile.mkstemp(prefix="autoedit_prev_", suffix=".wav")
            os.close(fd)
            self.current_preview_file = temp_preview
        except Exception:
            temp_preview = os.path.join(tempfile.gettempdir(), f"autoedit_prev_{int(time.time() * 1000)}.wav")
            self.current_preview_file = temp_preview

        extracted_ok = False
        audio_source_path = self.audio_path
        try:
            cmd = [
                self.ffmpeg_exe, "-y",
                "-ss", f"{start_t:.3f}",
                "-i", audio_source_path,
                "-t", f"{dur:.3f}",
                "-vn",
                "-c:a", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                temp_preview
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(temp_preview) and os.path.getsize(temp_preview) > 44:
                extracted_ok = True
            else:
                cmd2 = [
                    self.ffmpeg_exe, "-y",
                    "-i", audio_source_path,
                    "-ss", f"{start_t:.3f}",
                    "-t", f"{dur:.3f}",
                    "-vn",
                    "-c:a", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    temp_preview
                ]
                res2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res2.returncode == 0 and os.path.exists(temp_preview) and os.path.getsize(temp_preview) > 44:
                    extracted_ok = True
        except Exception:
            extracted_ok = False

        try:
            if extracted_ok:
                pygame.mixer.music.load(temp_preview)
            else:
                pygame.mixer.music.load(self.audio_path)
                try:
                    pygame.mixer.music.set_pos(start_t)
                except Exception:
                    pass

            pygame.mixer.music.play()
            self.is_preview_playing = True
            self._preview_start_t = float(start_t)
            self._preview_dur = float(dur)
            self._preview_offset = float(time.perf_counter())
            self.btn_audio_preview.configure(text="Pause", fg_color="#c026d3", hover_color="#9333ea")

            def check_finished():
                if not self.is_preview_playing:
                    return
                if not pygame.mixer.music.get_busy():
                    self.stop_audio_preview()
                    return

                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms is not None and pos_ms >= 0:
                    pos = self._preview_start_t + (pos_ms / 1000.0)
                else:
                    pos = self._preview_start_t + (time.perf_counter() - self._preview_offset)

                pos = min(self._preview_start_t + self._preview_dur, pos)
                self._draw_playhead(pos)
                self.lbl_playback_time.configure(
                    text=f"{pos:.1f}s / {self._preview_start_t + self._preview_dur:.1f}s"
                )
                self.after(50, check_finished)

            self.after(50, check_finished)
        except Exception as e:
            self.show_error("Preview-Fehler", f"Konnte Audio nicht abspielen:\n{os.path.basename(self.audio_path)}\n\n{e}")
            self.stop_audio_preview()

    def stop_audio_preview(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass
        if getattr(self, "current_preview_file", None) and os.path.exists(self.current_preview_file):
            try:
                os.remove(self.current_preview_file)
            except Exception:
                pass
            self.current_preview_file = None
        self.is_preview_playing = False
        self.clear_playhead()
        self.lbl_playback_time.configure(text="")
        self.btn_audio_preview.configure(text="Vorhören", fg_color="#2b719e", hover_color="#1f5375")

    # =====================================================================
    # Cancellation & Post-Actions
    # =====================================================================
    def cancel_generation(self):
        self.is_cancelled = True
        self.set_status("Abbruch angefordert... Bereinige Prozesse.", 0.0)

    def open_output_folder(self):
        if self.last_rendered_path and os.path.exists(self.last_rendered_path):
            folder = os.path.dirname(self.last_rendered_path)
            subprocess.Popen(f'explorer /select,"{os.path.normpath(self.last_rendered_path)}"')
        else:
            self.show_error("Fehler", "Kein erstelltes Video gefunden.")

    def play_output_video(self):
        if self.last_rendered_path and os.path.exists(self.last_rendered_path):
            os.startfile(self.last_rendered_path)
        else:
            self.show_error("Fehler", "Video-Datei existiert nicht.")

    # =====================================================================
    # Generation Workflow
    # =====================================================================
    def _render_busy(self):
        return str(self.btn_generate.cget("state")) == "disabled"

    def _cuts_unusable_message(self, action):
        problem = self.manual_cuts_problem() or "Die gespeicherten Cuts sind nicht verwendbar."
        extra = ""
        if "Cuts liegen zwischen" in problem:
            extra = ("\n\nMit 'Bereich laden' in Punkt 3 stellst du Start und Ende auf den "
                     "aufgezeichneten Bereich zurueck.")
        elif "Checkbox" in problem:
            extra = "\n\nSetze das Haekchen 'Beim Rendern verwenden' in Punkt 3."
        return f"{action} kann mit deinen Cuts gerade nicht laufen:\n\n{problem}{extra}"

    def start_generation(self):
        if self._render_busy():
            return
        inputs = self._collect_render_inputs()
        if not inputs:
            return
        start_t, end_t = inputs

        mode = self.mode_var.get()
        saved_cuts = self.effective_manual_cuts()

        if not saved_cuts and mode == "manual" and len(self.manual_cuts) >= 2:
            # Cuts sind da, greifen aber gerade nicht: nicht stillschweigend
            # eine neue Aufnahme starten - das sah aus, als waere alles verloren.
            self.show_info("Cuts vorhanden, aber nicht verwendbar",
                           self._cuts_unusable_message("Der Render"))
            return

        if not saved_cuts and mode == "manual":
            if not messagebox.askokcancel(
                "Manueller Modus",
                "Im manuellen Modus tippst du die Schnitte selbst zur Musik.\n\n"
                "Gleich startet die Aufnahme. Nach dem Pruefen der Cuts wird das "
                "Video automatisch gerendert."
            ):
                return
            out_path = self._ask_output_path("Video speichern als")
            if not out_path:
                return
            self.start_manual_recording(render_after=out_path)
            return

        out_path = self._ask_output_path("Video speichern als")
        if not out_path:
            return
        self._launch_render(start_t, end_t, out_path, mode, proxy=False)

    def _launch_render(self, start_t, end_t, out_path, mode, proxy=False):
        """Startet genau einen Render im Hintergrund (Haupt-Render oder Vorschau)."""
        self.is_cancelled = False
        self._set_render_buttons(True)
        self.post_action_frame.pack_forget()
        self.set_eta("")
        render_settings = self.collect_render_settings()
        cuts = render_settings.get("manual_cuts")
        if cuts:
            self.set_status(f"Starte Render mit {max(0, len(cuts) - 2)} gespeicherten Cuts...", 0.03)
            mode = "manual"
        else:
            self.set_status("Starte Beat-Analyse & Vorbereitung...", 0.03)
        threading.Thread(
            target=self.generate_edit,
            args=(start_t, end_t, out_path),
            kwargs={"mode": mode, "proxy": proxy, "settings": render_settings},
            daemon=True
        ).start()

    def _collect_render_inputs(self):
        if not self.video_paths:
            self.show_error("Fehler", "Bitte wähle mindestens ein Video oder Bild aus.")
            return None
        missing = [p for p in self.video_paths if not os.path.exists(p)]
        if missing:
            self.show_error(
                "Fehler",
                f"{len(missing)} Mediendatei(en) existieren nicht mehr, z. B.:\n"
                f"{missing[0]}\n\nBitte aus der Liste entfernen."
            )
            return None
        if not self.audio_path or not os.path.exists(self.audio_path):
            self.show_error("Fehler", "Bitte wähle eine Musik-Datei aus.")
            return None
        start_t = float(self.slider_start.get())
        end_t = float(self.slider_end.get())
        if end_t - start_t < 1.0:
            self.show_error("Fehler", "Der Musik-Bereich muss mindestens eine Sekunde lang sein.")
            return None
        return start_t, end_t

    def collect_render_settings(self):
        """
        Nimmt alle Render-Optionen als einfachen Snapshot auf dem UI-Thread auf.
        So liest der Render-Worker keine Tk-Variablen aus einem Fremd-Thread.
        """
        videos = list(self.video_paths)
        if self.shuffle_var.get():
            # Nur die Render-Kopie mischen - die Liste des Users bleibt wie sie ist.
            random.shuffle(videos)
        return {
            "audio_path": self.audio_path,
            "video_paths": videos,
            "overlay_img_path": self.overlay_img_path,
            "manual_cuts": self.effective_manual_cuts(),
            "orientation": self.orient_var.get(),
            "scale_mode": self.scale_mode_var.get(),
            "pace_choice": self.pace_var.get(),
            "fps": int(self.fps_var.get()),
            "clip_audio_setting": self.clip_audio_var.get(),
            "enable_fade_out": self.fade_out_var.get(),
            "apply_cinematic_lut": self.fx_cinematic_lut.get(),
            "enable_auto_loudness": self.auto_loudness_var.get(),
            "quality_str": self.quality_var.get(),
            "res_choice": self.res_var.get(),
            "fx_intensity": float(self.slider_fx_intensity.get()) / 100.0,
            "allow_shake": self.fx_shake.get(),
            "allow_bloom": self.fx_bloom_flash.get(),
            "allow_punch": self.fx_punch_zoom.get(),
            "allow_vignette": self.fx_vignette.get(),
            "allow_speedramp": self.fx_speedramp.get(),
            "fx_cadence": self.fx_cadence_var.get(),
            "effect_set": self.effect_set_var.get(),
            "fx_grain": self.fx_grain.get(),
            "fx_collage": self.fx_collage.get(),
            "fx_spin": self.fx_spin.get(),
            "fx_zoom_blur": self.fx_zoom_blur.get(),
            "enable_snap": self.snap_jitter_var.get(),
            "preroll_choice": self.preroll_var.get(),
            "ai_audio_analysis": self.ai_audio_analysis_var.get(),
            "ai_reframe": self.ai_reframe_var.get(),
            "enable_overlay": self.enable_overlay_var.get(),
            "overlay_start_type": self.overlay_start_mode.get(),
            "overlay_corner_mode": self.overlay_corner_var.get(),
            "overlay_size_str": self.overlay_size_var.get(),
            "overlay_speed_val": float(self.slider_ov_speed.get()),
            "overlay_opacity_val": float(self.slider_ov_opacity.get()) / 100.0,
            "overlay_circular": self.overlay_circle_cut_var.get(),
            "overlay_manual_sec": float(self.slider_overlay_sec.get()),
        }

    def _set_render_buttons(self, busy):
        state = "disabled" if busy else "normal"
        for btn in (self.btn_generate, self.btn_proxy, self.btn_batch,
                    getattr(self, "btn_record", None)):
            if btn is not None:
                btn.configure(state=state)
        self.btn_cancel.configure(state="normal" if busy else "disabled")

    def start_proxy_render(self):
        if self._render_busy():
            return
        inputs = self._collect_render_inputs()
        if not inputs:
            return
        start_t, end_t = inputs

        mode = self.mode_var.get()
        if mode == "manual" and not self.effective_manual_cuts():
            self.show_info("Hinweis", self._cuts_unusable_message("Die Schnell-Vorschau")
                           if len(self.manual_cuts) >= 2 else
                           "Im manuellen Modus braucht die Vorschau aufgenommene Cuts.\n\n"
                           "Nutze 'Jetzt aufnehmen' in Punkt 3.")
            return

        # Vorschau landet im Temp-Ordner und oeffnet sich direkt - kein Dialog.
        out_path = os.path.join(tempfile.gettempdir(), f"autoedit_preview_{int(time.time())}.mp4")
        self._launch_render(start_t, end_t, out_path, mode, proxy=True)

    def start_batch_export(self):
        if self._render_busy():
            return
        inputs = self._collect_render_inputs()
        if not inputs:
            return
        start_t, end_t = inputs

        mode = self.mode_var.get()
        if mode == "manual" and not self.effective_manual_cuts():
            self.show_info("Hinweis", self._cuts_unusable_message("Der Batch-Export")
                           if len(self.manual_cuts) >= 2 else
                           "Im manuellen Modus braucht der Batch-Export aufgenommene Cuts.\n\n"
                           "Nutze 'Jetzt aufnehmen' in Punkt 3.")
            return

        dialog = ctk.CTkInputDialog(
            title="Batch-Export",
            text="Wie viele Varianten rendern? (1-50, jede mit neu gemischter Clip-Reihenfolge)"
        )
        raw = dialog.get_input()
        if not raw:
            return
        try:
            count = int(str(raw).strip())
        except ValueError:
            self.show_error("Fehler", "Bitte eine ganze Zahl eingeben.")
            return
        count = max(1, min(50, count))

        folder = filedialog.askdirectory(
            title="Zielordner für Batch-Export wählen",
            initialdir=self._last_dir("last_output_dir") or None
        )
        if not folder:
            return
        self._remember_dir("last_output_dir", folder)

        base_name = os.path.splitext(os.path.basename(self.audio_path))[0][:40] or "edit"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.is_cancelled = False
        self.batch_running = True
        self.post_action_frame.pack_forget()
        self.set_eta("")
        self._set_render_buttons(True)
        render_settings = self.collect_render_settings()
        if render_settings.get("manual_cuts"):
            mode = "manual"

        def _worker():
            done = 0
            last_out = None
            try:
                for i in range(count):
                    if self.is_cancelled:
                        break
                    shuffled = list(render_settings["video_paths"])
                    random.shuffle(shuffled)
                    variant_settings = dict(render_settings)
                    variant_settings["video_paths"] = shuffled
                    variant_settings["variant"] = i
                    out_path = os.path.join(folder, f"{base_name}_{stamp}_v{i + 1:02d}.mp4")
                    ok = self.generate_edit(
                        start_t, end_t, out_path, mode=mode, proxy=False,
                        batch_label=f"Batch {i + 1}/{count}",
                        settings=variant_settings
                    )
                    if ok:
                        done += 1
                        last_out = out_path
            finally:
                self.batch_running = False
                self.set_eta("")
                if self.is_cancelled:
                    self.set_status(f"Batch abgebrochen ({done}/{count} fertig).", 0.0)
                else:
                    self.set_status(f"Batch-Export fertig: {done}/{count} Videos in {folder}", 1.0)
                    if last_out:
                        self.show_render_done(last_out, extra_msg=f"{done} Videos im Ordner:\n{folder}")
                self._ui_after(0, lambda: self._set_render_buttons(False))

        threading.Thread(target=_worker, daemon=True).start()

    # =====================================================================
    # Latency Calibration (manueller Beat-Recorder)
    # =====================================================================
    def run_latency_calibration(self, parent, current_ms=70.0, on_result=None):
        """
        Misst die echte Reaktions-/Audio-Latenz: es werden 8 Klicks
        abgespielt, der Nutzer tippt mit, und aus der Differenz zwischen
        Klickzeit und Tap-Zeit wird der Median als Offset verwendet.
        """
        CLICK_COUNT = 12
        INTERVAL = 0.55
        LEAD_IN = 1.2

        win = ctk.CTkToplevel(parent)
        win.title("Latenz kalibrieren")
        win.resizable(False, False)
        win.transient(parent)
        win.grab_set()

        ctk.CTkLabel(
            win,
            text="Latenz-Kalibrierung",
            font=("Arial", 17, "bold")
        ).pack(pady=(16, 4))

        ctk.CTkLabel(
            win,
            text="Klicke auf JEDEN Klick genau mit der Leertaste\n(oder der Maustaste) - so wie du spaeter schneidest.",
            font=("Arial", 12),
            text_color="#94a3b8",
            justify="center"
        ).pack(pady=(0, 10))

        lbl_state = ctk.CTkLabel(win, text="Bereit.", font=("Arial", 13))
        lbl_state.pack(pady=(0, 4))

        lbl_dots = ctk.CTkLabel(win, text="", font=("Consolas", 16, "bold"), text_color="#facc15")
        lbl_dots.pack(pady=(0, 8))

        lbl_result = ctk.CTkLabel(win, text="", font=("Arial", 20, "bold"), text_color="#22c55e")
        lbl_result.pack(pady=(0, 6))

        fine_var = ctk.DoubleVar(value=float(current_ms))
        lbl_fine = ctk.CTkLabel(win, text=f"Feinjustierung: {current_ms:.0f} ms", font=("Arial", 11))
        lbl_fine.pack()

        def _fine_cb(v):
            lbl_fine.configure(text=f"Feinjustierung: {v:.0f} ms")

        ctk.CTkSlider(
            win, from_=0, to=400, variable=fine_var,
            command=_fine_cb
        ).pack(fill="x", padx=60, pady=(2, 8))

        click_sound = build_metronome_click()
        if click_sound is None:
            ctk.CTkLabel(
                win, text="Kein Klick-Sound moeglich - bitte Wert manuell einstellen.",
                text_color="#ef4444", font=("Arial", 11)
            ).pack(pady=4)

        state = {"running": False, "t0": 0.0, "clicks": [], "taps": [], "timers": []}
        result = {"ms": float(current_ms)}

        def _cleanup():
            for t in state["timers"]:
                try:
                    win.after_cancel(t)
                except Exception:
                    pass
            state["timers"] = []

        def _finish():
            state["running"] = False
            _cleanup()

            used_clicks = set()
            offsets = []
            for tap in state["taps"]:
                best_i, best_d = None, 1e9
                for i, c in enumerate(state["clicks"]):
                    if i in used_clicks:
                        continue
                    d = abs(tap - c)
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i is not None and best_d <= 0.5:
                    used_clicks.add(best_i)
                    # Positiver Wert = Nutzer kommt spaeter als der Klick (Latenz)
                    offsets.append(tap - state["clicks"][best_i])

            if len(offsets) >= 2:
                med = max(0.0, float(np.median(offsets)))
                jitter = float(np.std(offsets)) if len(offsets) > 1 else 0.0
                fine_var.set(max(0.0, min(400.0, med * 1000.0)))
                _fine_cb(max(0.0, min(400.0, med * 1000.0)))
                lbl_result.configure(
                    text=f"{med * 1000.0:.0f} ms", text_color="#22c55e"
                )
                lbl_state.configure(
                    text=f"{len(offsets)} von {CLICK_COUNT} Klicks erkannt | Streuung {jitter * 1000:.0f} ms",
                    text_color="#94a3b8"
                )
            else:
                lbl_result.configure(text="nicht genug Taps", text_color="#ef4444")
                lbl_state.configure(text="Bitte wiederholen und ALLE Klicks erwischen.", text_color="#ef4444")

        def _schedule(i):
            if i >= CLICK_COUNT:
                win.after(int((INTERVAL * 1000) * 0.35), _finish)
                return
            when = (state["t0"] + LEAD_IN + i * INTERVAL) - time.perf_counter()
            if when < 0:
                when = 0.0
            timer = win.after(
                int(when * 1000),
                lambda i=i: (state["clicks"].append(time.perf_counter()),
                             click_sound.play() if click_sound else None,
                             lbl_dots.configure(text="#" * (i + 1)),
                             _schedule(i + 1))
            )
            state["timers"].append(timer)

        def _on_tap(event=None):
            if not state["running"]:
                return
            state["taps"].append(time.perf_counter())
            lbl_state.configure(text=f"{len(state['taps'])} Taps erfasst...", text_color="#facc15")

        for key in ("<space>", "<Left>", "<Right>", "<Up>", "<Down>"):
            win.bind(key, _on_tap)
        win.bind("<Button-1>", _on_tap)
        win.bind("<Button-3>", _on_tap)

        def _start():
            _cleanup()
            state["clicks"] = []
            state["taps"] = []
            state["running"] = True
            state["t0"] = time.perf_counter()
            lbl_dots.configure(text="")
            lbl_result.configure(text="", text_color="#22c55e")
            lbl_state.configure(text="Warte auf Start...", text_color="#facc15")
            btn_start.configure(state="disabled")
            _schedule(0)

        def _apply():
            value = float(fine_var.get())
            result["ms"] = value
            self.set_manual_latency_ms(value)
            if on_result:
                on_result(value)
            win.destroy()

        btn_start = ctk.CTkButton(
            win, text="Kalibrierung starten", command=_start,
            fg_color="#7c3aed", hover_color="#5b21b6", height=40, width=220,
            font=("Arial", 14, "bold")
        )
        btn_start.pack(pady=(6, 8))

        ctk.CTkButton(
            win, text="Wert uebernehmen", command=_apply,
            fg_color="#1f8a4c", hover_color="#176b3b", height=38, width=220
        ).pack(pady=(0, 6))

        ctk.CTkButton(
            win, text="Abbrechen", command=lambda: (_cleanup(), win.destroy()),
            fg_color="#444444", hover_color="#555555", height=32, width=160
        ).pack(pady=(0, 12))

        win.protocol("WM_DELETE_WINDOW", lambda: (_cleanup(), win.destroy()))
        self._center_window(win, min_w=500, min_h=400)
        return win, result

    # =====================================================================
    # Aufnahme-Review: Player mit Waveform + Hit-Indikatoren
    # =====================================================================
    def open_manual_review(self, start_t, end_t, final_beats, raw_taps, snapped, loose,
                           master_audio_wav, onsets=None, render_after=None):
        """
        Zeigt die Aufnahme zur Kontrolle: Waveform mit einem Marker pro Cut,
        Playback mit Klick-Indikator an jedem Hit, Cuts verschieben, setzen
        und loeschen. 'Cuts uebernehmen' legt sie in der Sammlung ab und
        rendert direkt, falls die Aufnahme ueber 'Video Rendern' kam.
        """
        duration = max(0.5, end_t - start_t)
        beats_list = sorted(float(b) for b in final_beats)   # in-place editierbar
        if not beats_list or beats_list[0] > 1e-6:
            beats_list.insert(0, 0.0)
        if beats_list[-1] < duration - 1e-6:
            beats_list.append(duration)
        beats_list[0] = 0.0
        beats_list[-1] = duration
        onset_list = list(onsets) if onsets is not None else []
        temp_dir = os.path.dirname(master_audio_wav)

        win = ctk.CTkToplevel(self)
        win.title("Cuts prüfen & bearbeiten")
        win.minsize(820, 520)
        win.transient(self)

        ctk.CTkLabel(
            win, text="Passen die Cuts?", font=("Arial", 18, "bold")
        ).pack(pady=(14, 2))

        src_name = os.path.basename(self.manual_review_source or self.audio_path or "")
        ctk.CTkLabel(
            win,
            text=f"{src_name}  |  {start_t:.1f}s - {end_t:.1f}s  ({duration:.1f}s)",
            font=("Arial", 11), text_color="#38bdf8"
        ).pack(pady=(0, 2))

        lbl_summary = ctk.CTkLabel(win, text="", font=("Arial", 12), text_color="#94a3b8")
        lbl_summary.pack(pady=(0, 2))

        ctk.CTkLabel(
            win,
            text="Ziehen = Cut verschieben  |  Klick ins Leere = Abspielposition  |  Rechtsklick = Cut setzen/loeschen  |  "
                 "Entf = markierten Cut loeschen  |  Pfeile = 1 ms (Shift 10 ms)  |  Leertaste = Play  |  Mausrad = Zoom",
            font=("Arial", 10), text_color="#64748b", wraplength=820, justify="center"
        ).pack(pady=(0, 8))

        # --- Waveform ------------------------------------------------
        wave_wrap = ctk.CTkFrame(win, fg_color="transparent")
        wave_wrap.pack(fill="x", padx=16)

        wave = Canvas(wave_wrap, height=150, bg="#0b1220", highlightthickness=1,
                      highlightbackground="#1e293b", highlightcolor="#38bdf8",
                      takefocus=1)
        wave.pack(fill="x")

        lbl_status = ctk.CTkLabel(win, text="0.000s / 0.000s", font=("Consolas", 12), text_color="#facc15")
        lbl_status.pack(pady=(4, 0))

        state = {
            "env": None, "hop": 0.02, "playing": False, "t0": 0.0, "play_from": 0.0,
            "pos": 0.0, "next": 0, "indicators": True, "job": None,
            "drag_idx": None, "last_click": -1.0,
            "view": (0.0, duration), "sel": None, "pan_x": 0,
            "pcm": None, "channel": None, "closed": False,
        }

        def update_summary():
            lbl_summary.configure(
                text=f"{len(beats_list) - 2} Schnitte  |  {snapped} eingerastet  |  {loose} frei gesetzt"
            )

        update_summary()

        def wave_width():
            w = wave.winfo_width()
            if w <= 2:
                w = wave.winfo_reqwidth()
            return max(300, w)

        def wave_height():
            h = wave.winfo_height()
            return h if h > 2 else 150

        def view_span():
            return max(0.05, state["view"][1] - state["view"][0])

        def movable_range():
            return range(1, max(1, len(beats_list) - 1))

        def time_from_x(x):
            v0, v1 = state["view"]
            return max(0.0, min(duration, v0 + (x / float(wave_width())) * (v1 - v0)))

        def x_from_time(t):
            v0, v1 = state["view"]
            return int((t - v0) / max(1e-6, v1 - v0) * wave_width())

        def draw_cuts():
            """Nur die Cut-Marker neu zeichnen (fuer Draggen ohne Full-Redraw)."""
            wave.delete("cuts")
            w, h = wave_width(), wave_height()
            for idx in movable_range():
                b = beats_list[idx]
                x = x_from_time(b)
                if not (0 <= x < w):
                    continue
                active = (state["drag_idx"] == idx) or (state["sel"] == idx)
                col = "#facc15" if active else "#22c55e"
                wave.create_line(x, 0, x, h, fill=col, width=3 if active else 2, tags="cuts")
                wave.create_oval(x - 5, 2, x + 5, 12, fill=col, outline="", tags="cuts")
                if active or view_span() < duration * 0.6:
                    wave.create_text(x + 8, 9, text=f"{b:.3f}s", fill=col,
                                     font=("Consolas", 8, "bold"), anchor="w", tags="cuts")

        def draw_static():
            wave.delete("base")
            w, h = wave_width(), wave_height()
            mid = h / 2.0
            v0, v1 = state["view"]
            span = max(1e-6, v1 - v0)
            env = state["env"]
            if env is not None and len(env) > 0:
                max_v = float(np.percentile(env, 98)) or 1.0
                top, bot = [], []
                for x in range(0, w, 1):
                    i0 = int((v0 + (x / float(w)) * span) / state["hop"])
                    i1 = int((v0 + ((x + 1) / float(w)) * span) / state["hop"]) + 1
                    i0 = max(0, min(len(env) - 1, i0))
                    i1 = max(i0 + 1, min(len(env), i1))
                    seg = env[i0:i1]
                    v = (float(np.max(seg)) / max_v) if len(seg) else 0.0
                    amp = min(1.0, v) * (mid - 14)
                    top.extend((x, mid - amp))
                    bot.extend((x, mid + amp))
                if top:
                    wave.create_polygon(top + list(reversed(bot)), fill="#1e3a5f",
                                        outline="", tags="base")
            else:
                wave.create_text(w // 2, h // 2, text="Wellenform wird geladen...",
                                 fill="#475569", font=("Arial", 10), tags="base")
            for tap in raw_taps:
                x = int((tap - v0) / span * w)
                if 0 <= x < w:
                    wave.create_line(x, mid - 12, x, mid + 12, fill="#f59e0b", tags="base")
            wave.create_text(
                6, h - 9, text=f"{v0:.2f}s - {v1:.2f}s",
                fill="#475569", font=("Consolas", 9), anchor="w", tags="base"
            )
            wave.tag_lower("base")
            draw_cuts()

        def draw_cursor():
            wave.delete("cur")
            w, h = wave_width(), wave_height()
            x = x_from_time(state["pos"])
            if 0 <= x < w:
                wave.create_line(x, 0, x, h, fill="#facc15", width=2, tags="cur")

        def redraw():
            draw_static()
            draw_cursor()

        def set_pos(p):
            state["pos"] = max(0.0, min(duration, p))
            lbl_status.configure(text=f"{state['pos']:.3f}s / {duration:.3f}s")
            draw_cursor()

        # --- Zoom & Pan -------------------------------------------------
        def set_view(v0, v1):
            span = max(0.08, min(duration, v1 - v0))
            v0 = max(0.0, min(duration - span, v0))
            state["view"] = (v0, v0 + span)
            redraw()

        def on_wheel(event):
            v0, v1 = state["view"]
            span = v1 - v0
            t = time_from_x(event.x)
            factor = 0.8 if event.delta > 0 else 1.25
            new_span = max(0.08, min(duration, span * factor))
            ratio = (t - v0) / span if span > 0 else 0.5
            set_view(t - ratio * new_span, t - ratio * new_span + new_span)

        def on_pan_start(event):
            state["pan_x"] = event.x

        def on_pan(event):
            v0, v1 = state["view"]
            span = v1 - v0
            d = -(event.x - state["pan_x"]) / float(wave_width()) * span
            state["pan_x"] = event.x
            set_view(v0 + d, v0 + d + span)

        def reset_view():
            set_view(0.0, duration)

        # --- Cuts bearbeiten ----------------------------------------------
        def snap_to_onset(t, lo, hi):
            """Auf den naechsten echten Onset einrasten - nur wenn gewuenscht."""
            if not snap_var.get() or not onset_list:
                return max(lo, min(hi, t))
            best, best_d = None, 0.09
            for o in onset_list:
                if o < lo or o > hi:
                    continue
                d = abs(o - t)
                if d < best_d:
                    best, best_d = o, d
            return best if best is not None else max(lo, min(hi, t))

        def play_tick_sound(t):
            if click_sound is not None and abs(t - state["last_click"]) > 0.12:
                state["last_click"] = t
                click_sound.play()

        def move_cut(idx, t):
            lo = beats_list[idx - 1] + 0.10
            hi = beats_list[idx + 1] - 0.10
            if hi < lo:
                return
            t = max(lo, min(hi, t))
            beats_list[idx] = t
            draw_cuts()
            lbl_status.configure(text=f"Cut #{idx} -> {t:.3f}s")
            play_tick_sound(t)

        def nearest_cut(x):
            tol = max(9, int(wave_width() * 0.012))
            best_idx, best_d = None, tol
            for idx in movable_range():
                d = abs(x_from_time(beats_list[idx]) - x)
                if d < best_d:
                    best_idx, best_d = idx, d
            return best_idx

        def on_press(event):
            wave.focus_set()
            idx = nearest_cut(event.x)
            state["sel"] = idx
            if idx is not None:
                state["drag_idx"] = idx
                wave.configure(cursor="sb_h_double_arrow")
                draw_cuts()
                return
            draw_cuts()
            # Klick ins Leere: Abspielposition setzen (laufende Wiedergabe springt mit)
            t = time_from_x(event.x)
            if state["playing"]:
                start_playback(t)
            else:
                set_pos(t)

        def on_motion(event):
            idx = state["drag_idx"]
            if idx is None:
                return
            raw_t = time_from_x(event.x)
            t = snap_to_onset(raw_t, beats_list[idx - 1] + 0.02, beats_list[idx + 1] - 0.02)
            move_cut(idx, t)

        def on_release(event):
            if state["drag_idx"] is None:
                return
            state["drag_idx"] = None
            wave.configure(cursor="hand2")
            draw_cuts()
            if state["sel"] is not None:
                lbl_status.configure(
                    text=f"Cut #{state['sel']}: {beats_list[state['sel']]:.3f}s  (Pfeile = 1 ms, Shift = 10 ms, Entf = loeschen)"
                )

        def on_key(event):
            idx = state["sel"]
            if idx is None or not (1 <= idx < len(beats_list) - 1):
                return "break"
            step = 0.010 if (event.state & 0x0001) else 0.001   # Shift = 10ms
            delta = -step if event.keysym == "Left" else step
            move_cut(idx, beats_list[idx] + delta)
            return "break"

        def delete_cut(idx):
            if idx is None or not (1 <= idx < len(beats_list) - 1):
                return
            removed = beats_list.pop(idx)
            state["sel"] = None
            state["drag_idx"] = None
            update_summary()
            draw_cuts()
            lbl_status.configure(text=f"Cut bei {removed:.3f}s geloescht")

        def on_delete(event=None):
            delete_cut(state["sel"])
            return "break"

        def on_right_click(event):
            wave.focus_set()
            idx = nearest_cut(event.x)
            if idx is not None:
                delete_cut(idx)
                return
            t = time_from_x(event.x)
            t = snap_to_onset(t, 0.10, duration - 0.10)
            if t <= 0.10 or t >= duration - 0.10:
                return
            if any(abs(t - b) < 0.10 for b in beats_list):
                lbl_status.configure(text="Zu nah an einem bestehenden Cut (min. 100 ms Abstand)")
                return
            beats_list.append(t)
            beats_list.sort()
            state["sel"] = beats_list.index(t)
            update_summary()
            draw_cuts()
            lbl_status.configure(text=f"Cut gesetzt bei {t:.3f}s")
            play_tick_sound(t)

        snap_var = ctk.BooleanVar(value=False)
        wave.bind("<MouseWheel>", on_wheel)
        wave.bind("<ButtonPress-2>", on_pan_start)
        wave.bind("<B2-Motion>", on_pan)
        wave.bind("<Double-Button-1>", lambda e: reset_view())
        wave.bind("<ButtonPress-1>", on_press)
        wave.bind("<B1-Motion>", on_motion)
        wave.bind("<ButtonRelease-1>", on_release)
        wave.bind("<Button-3>", on_right_click)
        for key in ("<Left>", "<Right>", "<Shift-Left>", "<Shift-Right>"):
            wave.bind(key, on_key)
        wave.bind("<Delete>", on_delete)
        wave.bind("<BackSpace>", on_delete)
        wave.configure(cursor="hand2")

        # --- Wiedergabe (seekbar: Sound aus dem PCM-Puffer ab Position) ---
        click_sound = build_metronome_click()

        def load_pcm():
            if state["pcm"] is None:
                snd = pygame.mixer.Sound(master_audio_wav)
                arr = pygame.sndarray.array(snd)
                if arr.ndim == 1:
                    arr = np.stack([arr, arr], axis=1)
                state["pcm"] = np.ascontiguousarray(arr)
            return state["pcm"]

        def stop_playback():
            state["playing"] = False
            if state["job"] is not None:
                try:
                    win.after_cancel(state["job"])
                except Exception:
                    pass
                state["job"] = None
            if state["channel"] is not None:
                try:
                    state["channel"].stop()
                except Exception:
                    pass
                state["channel"] = None
            if not state["closed"]:
                btn_play.configure(text="Abspielen", fg_color="#1f8a4c", hover_color="#176b3b")

        def tick():
            if not state["playing"]:
                return
            now = state["play_from"] + (time.perf_counter() - state["t0"])
            if now >= duration:
                stop_playback()
                set_pos(0.0)
                return
            # Audio-Indikator an jedem Cut
            if state["indicators"] and click_sound is not None:
                while state["next"] < len(beats_list) and beats_list[state["next"]] <= now:
                    mark = beats_list[state["next"]]
                    if 0.05 <= mark <= duration - 0.05:
                        click_sound.play()
                    state["next"] += 1
            set_pos(now)
            state["job"] = win.after(15, tick)

        def start_playback(from_t=None):
            stop_playback()
            pos = state["pos"] if from_t is None else float(from_t)
            if pos >= duration - 0.05:
                pos = 0.0
            try:
                pcm = load_pcm()
                i0 = int(pos * 44100)
                sound = pygame.sndarray.make_sound(np.ascontiguousarray(pcm[i0:]))
                state["channel"] = sound.play()
                state["sound"] = sound
            except Exception as e:
                self.show_error("Fehler", f"Wiedergabe nicht moeglich:\n{e}")
                return
            state["playing"] = True
            state["play_from"] = pos
            state["t0"] = time.perf_counter()
            state["next"] = int(np.searchsorted(np.array(beats_list), pos + 1e-4))
            set_pos(pos)
            btn_play.configure(text="Stopp", fg_color="#8b1e1e", hover_color="#631414")
            state["job"] = win.after(15, tick)

        def toggle_play(event=None):
            if state["playing"]:
                stop_playback()
            else:
                start_playback()
            return "break"

        win.bind("<space>", toggle_play)

        # --- Transport ------------------------------------------------
        transport = ctk.CTkFrame(win, fg_color="transparent")
        transport.pack(fill="x", padx=16, pady=(10, 0))
        transport.grid_columnconfigure(5, weight=1)

        btn_play = ctk.CTkButton(
            transport, text="Abspielen", width=130, height=38, command=toggle_play,
            fg_color="#1f8a4c", hover_color="#176b3b", font=("Arial", 13, "bold")
        )
        btn_play.grid(row=0, column=0, padx=(0, 8))

        ctk.CTkButton(
            transport, text="Von vorn", width=90, height=38,
            command=lambda: start_playback(0.0) if state["playing"] else set_pos(0.0),
            fg_color="#334155", hover_color="#475569"
        ).grid(row=0, column=1, padx=(0, 8))

        ind_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            transport, text="Klick auf jedem Cut", variable=ind_var,
            command=lambda: state.update(indicators=ind_var.get())
        ).grid(row=0, column=2, padx=(6, 12), sticky="w")

        ctk.CTkCheckBox(
            transport, text="An Beats einrasten", variable=snap_var
        ).grid(row=0, column=3, padx=(0, 12), sticky="w")

        ctk.CTkButton(
            transport, text="Zoom zurücksetzen", command=reset_view,
            fg_color="#334155", hover_color="#475569", width=140, height=38
        ).grid(row=0, column=6, sticky="e")

        # --- Entscheidung ---------------------------------------------
        decision = ctk.CTkFrame(win, fg_color="transparent")
        decision.pack(fill="x", padx=16, pady=(12, 12))
        decision.grid_columnconfigure(0, weight=1)

        hint = ("'Cuts uebernehmen' speichert die Cuts und rendert das Video direkt."
                if render_after else
                "'Cuts uebernehmen' legt sie in der Sammlung (Punkt 3) ab - fuer Rendern, Vorschau und Batch.")
        ctk.CTkLabel(
            decision, text=hint, font=("Arial", 10), text_color="#94a3b8", anchor="w"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        def close_window():
            state["closed"] = True
            stop_playback()
            try:
                win.destroy()
            except Exception:
                pass

        def accept():
            if len(beats_list) < 3:
                if not messagebox.askyesno(
                    "Keine Cuts", "Es ist kein einziger Cut gesetzt. Trotzdem uebernehmen?", parent=win
                ):
                    return
            close_window()
            stored = self.store_manual_cuts(
                list(beats_list),
                audio_source=self.manual_review_source or self.audio_path,
                start_t=start_t, end_t=end_t,
                snapped=snapped, loose=loose
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
            self._set_render_buttons(False)
            if not stored:
                return
            n = len(beats_list) - 2
            if render_after:
                if not self.effective_manual_cuts():
                    self.show_info("Hinweis", self._cuts_unusable_message("Der Render"))
                    return
                self._launch_render(float(self.slider_start.get()), float(self.slider_end.get()),
                                    render_after, "manual", proxy=False)
                return
            self.set_status(
                f"{n} manuelle Cuts gespeichert ({snapped} am Beat eingerastet, {loose} frei) - "
                f"bereit fuer Rendern, Vorschau und Batch.", 0.0
            )

        def redo():
            close_window()
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.set_status("Aufnahme verworfen - neuer Versuch...", 0.0)
            self.open_manual_recorder(start_t, end_t, render_after=render_after)

        def discard():
            close_window()
            shutil.rmtree(temp_dir, ignore_errors=True)
            self._set_render_buttons(False)
            self.set_status("Review geschlossen - keine Aenderungen uebernommen.", 0.0)

        ctk.CTkButton(
            decision, text="Verwerfen", command=discard,
            fg_color="#444444", hover_color="#555555", width=120, height=42
        ).grid(row=1, column=0, sticky="e", padx=6)

        ctk.CTkButton(
            decision, text="Neu aufnehmen", command=redo,
            fg_color="#8b1e1e", hover_color="#631414", width=150, height=42
        ).grid(row=1, column=1, sticky="e", padx=6)

        ctk.CTkButton(
            decision, text="Cuts uebernehmen & rendern" if render_after else "Cuts uebernehmen",
            command=accept, fg_color="#1f8a4c", hover_color="#176b3b", width=240, height=42,
            font=("Arial", 14, "bold")
        ).grid(row=1, column=2, sticky="e", padx=6)

        wave.bind("<Configure>", lambda e: redraw())
        win.protocol("WM_DELETE_WINDOW", discard)

        def _load_wave():
            try:
                env, hop = compute_waveform_envelope(master_audio_wav, 0.0, duration)
            except Exception as e:
                print(f"[Review] Waveform nicht ladbar: {e}")
                env, hop = None, 0.02

            def _apply():
                if state["closed"]:
                    return
                state["env"] = env
                state["hop"] = hop
                redraw()

            self._ui_after(0, _apply)

        threading.Thread(target=_load_wave, daemon=True).start()
        self._center_window(win, min_w=900, min_h=540, max_w=1600)
        win.after(120, wave.focus_set)
        return win

    # =====================================================================
    # Wiederverwendbare Cut-Sammlung (manueller Modus)
    # =====================================================================
    def store_manual_cuts(self, cuts, audio_source=None, start_t=0.0, end_t=None,
                          snapped=0, loose=0):
        """Legt gepruefte Cut-Zeiten als wiederverwendbare Sammlung ab."""
        self._flush_history()
        cleaned = []
        for c in sorted(float(x) for x in cuts):
            if cleaned and c <= cleaned[-1] + 1e-4:
                continue
            cleaned.append(c)
        if len(cleaned) < 2:
            self.show_error("Hinweis", "Es wurden keine brauchbaren Cuts uebernommen.")
            return False
        self.manual_cuts = cleaned
        self.manual_cuts_audio = audio_source or self.audio_path
        self.manual_cuts_range = (float(start_t), float(end_t if end_t is not None else 0.0))
        self.manual_cuts_info = {"snapped": int(snapped), "loose": int(loose)}
        # Frisch aufgenommene Cuts sollen auch benutzt werden.
        self.use_manual_cuts_var.set(True)
        self.refresh_manual_cuts_ui()
        self.note_change()
        return True

    def clear_manual_cuts(self, confirm=False):
        if not self.manual_cuts:
            return
        if confirm and not messagebox.askyesno(
            "Cuts verwerfen?",
            f"{max(0, len(self.manual_cuts) - 2)} gespeicherte Cuts verwerfen?\n(Strg+Z holt sie zurueck)"
        ):
            return
        self._flush_history()
        self.manual_cuts = []
        self.manual_cuts_audio = None
        self.manual_cuts_range = None
        self.manual_cuts_info = {}
        self.use_manual_cuts_var.set(False)
        self.refresh_manual_cuts_ui()
        self.note_change()

    def _on_manual_cuts_toggle(self):
        self.refresh_manual_cuts_ui()
        self.note_change()

    def manual_cuts_absolute(self):
        """
        Die gespeicherten Cuts in absoluten Sekunden der Musikdatei.

        Der Recorder speichert relative Zeiten (0 = Start des Aufnahmefensters);
        fuer den Bereich-Vergleich braucht es absolute Zeiten - daher der
        gespeicherte Aufnahme-Start als Basis.
        """
        base = 0.0
        if self.manual_cuts_range:
            try:
                base = float(self.manual_cuts_range[0])
            except Exception:
                base = 0.0
        return [base + float(c) for c in self.manual_cuts]

    def manual_cuts_problem(self):
        """
        Leerer String, wenn die gespeicherten Cuts verwendbar sind - sonst der
        konkrete Grund.
        """
        if len(self.manual_cuts) < 2:
            return "Es sind noch keine Cuts gespeichert."
        if self.manual_cuts_audio and self.audio_path:
            a = os.path.normcase(os.path.abspath(self.manual_cuts_audio))
            b = os.path.normcase(os.path.abspath(self.audio_path))
            if a != b:
                return (f"Die Cuts gehoeren zu '{os.path.basename(self.manual_cuts_audio)}', "
                        f"geladen ist '{os.path.basename(self.audio_path)}'.")
        if not self.audio_duration:
            return "Es ist keine Musik geladen."
        if not self.use_manual_cuts_var.get():
            return "Die Checkbox 'Beim Rendern verwenden' ist nicht gesetzt."

        start_t = float(self.slider_start.get())
        end_t = float(self.slider_end.get())
        abs_cuts = self.manual_cuts_absolute()
        inside = [c for c in abs_cuts if start_t - 0.05 <= c <= end_t + 0.05]
        if len(inside) < 2:
            return (f"Die Cuts liegen zwischen {min(abs_cuts):.1f}s und {max(abs_cuts):.1f}s, "
                    f"gewaehlt ist aber {start_t:.1f}s bis {end_t:.1f}s.")
        return ""

    def apply_manual_cuts_range(self):
        """Setzt Start/Ende auf den Bereich, in dem aufgezeichnet wurde."""
        if not self.manual_cuts_range or not self.audio_duration:
            return
        try:
            a, b = float(self.manual_cuts_range[0]), float(self.manual_cuts_range[1])
        except Exception:
            return
        if b <= a:
            return
        self.set_range(start=a, end=b)

    def effective_manual_cuts(self):
        """
        Die Cuts, die ein Render tatsaechlich verwenden soll - oder None.

        Relativ zum Start des Renderfensters, immer mit 0.0 am Anfang und der
        Fensterlaenge am Ende. Cuts, die naeher als 100 ms an einer Grenze
        liegen, wuerden einen unsichtbaren Mini-Clip erzeugen und fallen weg.
        """
        if self.manual_cuts_problem():
            return None
        start_t = float(self.slider_start.get())
        end_t = float(self.slider_end.get())
        total = end_t - start_t
        rel = [c - start_t for c in self.manual_cuts_absolute()]
        inner = [round(c, 4) for c in rel if 0.10 <= c <= total - 0.10]
        return [0.0] + inner + [round(total, 4)]

    def refresh_manual_cuts_ui(self):
        lbl = getattr(self, "lbl_manual_cuts", None)
        if lbl is None:
            return
        n = len(self.manual_cuts)
        has_cuts = n >= 2
        for name in ("btn_manual_cuts_clear", "btn_manual_cuts_view"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.configure(state="normal" if has_cuts else "disabled")
        if not has_cuts:
            lbl.configure(
                text="Keine Cuts gespeichert - 'Jetzt aufnehmen' tippt die Schnitte live zur Musik.",
                text_color="#64748b"
            )
            if hasattr(self, "btn_manual_cuts_range"):
                self.btn_manual_cuts_range.configure(state="disabled")
            return

        base = os.path.basename(self.manual_cuts_audio) if self.manual_cuts_audio else "?"
        matches = bool(self.manual_cuts_audio and self.audio_path
                       and os.path.normcase(os.path.abspath(self.manual_cuts_audio))
                       == os.path.normcase(os.path.abspath(self.audio_path)))
        rng = self.manual_cuts_range or (0.0, 0.0)

        # Die Checkbox wird hier bewusst NICHT umgeschaltet: wer sie abwaehlt,
        # will die Cuts nicht nutzen - egal, was sich sonst aendert.
        problem = self.manual_cuts_problem()
        info = f"{n - 2} Cuts  |  {base}  |  aufgenommen {rng[0]:.1f}s - {rng[1]:.1f}s"
        if problem:
            info += f"\nNicht aktiv: {problem}"
        else:
            info += "\nAktiv: Rendern, Vorschau und Batch nutzen diese Cuts."
        lbl.configure(text=info, text_color="#22c55e" if not problem else "#f59e0b")
        if hasattr(self, "btn_manual_cuts_range"):
            self.btn_manual_cuts_range.configure(
                state="normal" if matches and self.audio_duration else "disabled"
            )

    def open_manual_recorder(self, start_t, end_t, render_after=None):
        self.manual_review_source = self.audio_path
        self._set_render_buttons(True)

        def _ready(temp_dir, wav, onsets):
            self._show_manual_recorder(start_t, end_t, temp_dir, wav, onsets, render_after)

        self._prepare_manual_audio(start_t, end_t, _ready)

    def _show_manual_recorder(self, start_t, end_t, temp_dir, master_audio_wav, detected_onsets, render_after):
        total_duration = end_t - start_t
        self.manual_times = []

        rec_win = ctk.CTkToplevel(self)
        rec_win.title("Live-Aufnahme: Schnitte tippen")
        rec_win.transient(self)

        ctk.CTkLabel(
            rec_win, text="Schnitte live zur Musik tippen", font=("Arial", 18, "bold")
        ).pack(pady=(14, 2))
        ctk.CTkLabel(
            rec_win,
            text=f"{os.path.basename(self.audio_path or '')}  |  {start_t:.1f}s - {end_t:.1f}s  ({total_duration:.1f}s)\n"
                 "LEERTASTE / Pfeiltasten oder Klick in die Flaeche = Schnitt.  "
                 "Esc oder 'Fertig' = Aufnahme beenden.",
            font=("Arial", 12), text_color="#94a3b8", justify="center"
        ).pack(pady=(0, 8))

        magnet_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            rec_win, text="Takt-Magnet (Taps auf erkannte Drum-Hits einrasten)", variable=magnet_var
        ).pack(pady=(0, 6))

        # --- Tap-Flaeche -----------------------------------------------------
        tap_area = ctk.CTkFrame(rec_win, height=150, fg_color="#0b1220", corner_radius=12,
                                border_width=2, border_color="#1e293b")
        tap_area.pack(fill="x", padx=18, pady=(4, 6))
        tap_area.pack_propagate(False)

        lbl_cuts = ctk.CTkLabel(tap_area, text="0 Schnitte", font=("Arial", 26, "bold"), text_color="#facc15")
        lbl_cuts.pack(pady=(26, 0))
        lbl_tap_feedback = ctk.CTkLabel(tap_area, text="Bereit - 'Aufnahme starten' druecken",
                                        font=("Consolas", 12, "bold"), text_color="#475569")
        lbl_tap_feedback.pack(pady=2)
        lbl_grid_hint = ctk.CTkLabel(tap_area, text="", font=("Arial", 11), text_color="#475569")
        lbl_grid_hint.pack(pady=(0, 2))

        progress = ctk.CTkProgressBar(rec_win, height=10)
        progress.pack(fill="x", padx=18, pady=(2, 2))
        progress.set(0.0)
        lbl_time = ctk.CTkLabel(rec_win, text=f"0.0s / {total_duration:.1f}s", font=("Consolas", 12), text_color="gray")
        lbl_time.pack(pady=(0, 4))

        # --- Latenz-Kalibrierung -------------------------------------------------
        lat_frame = ctk.CTkFrame(rec_win, fg_color="transparent")
        lat_frame.pack(fill="x", padx=18, pady=(6, 2))

        latency_var = ctk.DoubleVar(value=self.get_manual_latency_ms())
        lbl_latency = ctk.CTkLabel(
            lat_frame, text=f"Latenz-Korrektur: {self.get_manual_latency_ms():.0f} ms",
            width=190, anchor="w", font=("Arial", 11, "bold")
        )
        lbl_latency.pack(side="left")

        def _latency_cb(v):
            lbl_latency.configure(text=f"Latenz-Korrektur: {float(v):.0f} ms")

        latency_slider = ctk.CTkSlider(lat_frame, from_=0, to=400, variable=latency_var, command=_latency_cb)
        latency_slider.pack(side="left", fill="x", expand=True, padx=8)

        def _open_calibration():
            def _done(value):
                latency_var.set(value)
                _latency_cb(value)
                self.set_manual_latency_ms(value)

            cal_win, _ = self.run_latency_calibration(rec_win, current_ms=float(latency_var.get()), on_result=_done)
            try:
                rec_win.wait_window(cal_win)
                if rec_win.winfo_exists():
                    rec_win.grab_set()
                    rec_win.focus_force()
            except Exception:
                pass

        btn_calibrate = ctk.CTkButton(
            lat_frame, text="Kalibrieren", command=_open_calibration,
            fg_color="#7c3aed", hover_color="#5b21b6", width=110, height=26
        )
        btn_calibrate.pack(side="right")

        ctk.CTkLabel(
            rec_win,
            text="Tastatur, Soundkarte und Monitor haben eigene Verzoegerung. Einmal kalibrieren -\n"
                 "danach sitzen die Taps automatisch auf dem Beat.",
            font=("Arial", 10), text_color="#64748b", justify="center"
        ).pack(pady=(0, 4))

        try:
            pygame.mixer.music.load(master_audio_wav)
        except Exception as e:
            rec_win.destroy()
            shutil.rmtree(temp_dir, ignore_errors=True)
            self._set_render_buttons(False)
            self.show_error("Fehler", f"Audio konnte nicht geladen werden:\n{e}")
            return

        st = {"playing": False, "t0": 0.0, "finished": False, "job": None}
        onsets_arr = np.array(detected_onsets) if len(detected_onsets) > 0 else None

        def flash_area(color):
            tap_area.configure(border_color=color)
            rec_win.after(90, lambda: tap_area.winfo_exists() and tap_area.configure(border_color="#1e293b"))

        def record_cut(event=None):
            if not st["playing"]:
                return "break"
            latency = float(latency_var.get()) / 1000.0
            compensated_time = (time.perf_counter() - st["t0"]) - latency

            color = "#facc15"
            if onsets_arr is not None:
                nearest = float(onsets_arr[int(np.argmin(np.abs(onsets_arr - compensated_time)))])
                err_ms = (compensated_time - nearest) * 1000.0
                color = "#22c55e" if abs(err_ms) < 35 else ("#facc15" if abs(err_ms) < 90 else "#ef4444")
                lbl_tap_feedback.configure(text=f"Tap: {err_ms:+.0f} ms vom Beat", text_color=color)
            else:
                lbl_tap_feedback.configure(text="Tap erfasst", text_color=color)

            if 0.05 < compensated_time < total_duration:
                self.manual_times.append(compensated_time)
                lbl_cuts.configure(text=f"{len(self.manual_times)} Schnitt{'e' if len(self.manual_times) != 1 else ''}")
                flash_area(color)
            else:
                lbl_tap_feedback.configure(text="Tap ausserhalb des Bereichs", text_color="#ef4444")
            return "break"

        for key in ("<space>", "<Left>", "<Right>", "<Up>", "<Down>"):
            rec_win.bind(key, record_cut)
        for widget in (tap_area, lbl_cuts, lbl_tap_feedback, lbl_grid_hint):
            widget.bind("<Button-1>", record_cut)
            widget.bind("<Button-3>", record_cut)

        def update_clock():
            if not st["playing"]:
                return
            elapsed = time.perf_counter() - st["t0"]
            lbl_time.configure(text=f"{min(elapsed, total_duration):.1f}s / {total_duration:.1f}s")
            progress.set(max(0.0, min(1.0, elapsed / max(0.01, total_duration))))
            if onsets_arr is not None:
                pos = elapsed - float(latency_var.get()) / 1000.0
                idx = int(np.searchsorted(onsets_arr, pos))
                nxt = onsets_arr[idx] if idx < len(onsets_arr) else None
                prv = onsets_arr[idx - 1] if idx - 1 >= 0 else None
                cands = [c for c in (nxt, prv) if c is not None]
                if cands:
                    dist = min(abs(pos - c) for c in cands)
                    lbl_grid_hint.configure(
                        text=f"Naechster Hit in {dist * 1000:.0f} ms" if nxt is not None and abs(pos - nxt) == dist
                        else f"Letzter Hit vor {dist * 1000:.0f} ms",
                        text_color="#38bdf8" if dist < 0.12 else "#475569"
                    )
            if elapsed >= total_duration + 0.15:
                finish()
                return
            st["job"] = rec_win.after(30, update_clock)

        def start_play():
            if st["playing"]:
                return
            st["playing"] = True
            btn_start.configure(state="disabled", text="Aufnahme laeuft...")
            btn_calibrate.configure(state="disabled")
            lbl_tap_feedback.configure(text="Los! Tippe auf jeden Schnitt.", text_color="#38bdf8")
            st["t0"] = time.perf_counter()
            pygame.mixer.music.play()
            rec_win.focus_force()
            update_clock()

        def stop_audio():
            st["playing"] = False
            if st["job"] is not None:
                try:
                    rec_win.after_cancel(st["job"])
                except Exception:
                    pass
                st["job"] = None
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass

        def finish(event=None):
            if st["finished"]:
                return
            st["finished"] = True
            stop_audio()
            self.set_manual_latency_ms(float(latency_var.get()))
            try:
                rec_win.grab_release()
            except Exception:
                pass
            rec_win.destroy()

            if not self.manual_times:
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.set_status("Aufnahme beendet - keine Schnitte erfasst.", 0.0)
                self._set_render_buttons(False)
                return

            final_beats = []
            snapped = 0
            loose = 0
            raw_sorted = sorted(set(self.manual_times))
            if magnet_var.get() and onsets_arr is not None:
                snap_window = float(np.clip(float(latency_var.get()) / 1000.0 * 0.8 + 0.06, 0.06, 0.16))
                for tap in raw_sorted:
                    diffs = np.abs(onsets_arr - tap)
                    nearest_idx = int(np.argmin(diffs))
                    if diffs[nearest_idx] <= snap_window:
                        final_beats.append(float(onsets_arr[nearest_idx]))
                        snapped += 1
                    else:
                        final_beats.append(float(tap))
                        loose += 1
            else:
                final_beats = list(raw_sorted)
                loose = len(raw_sorted)

            # Doppelte/zu dichte Schnitte zusammenfassen (min. 100 ms)
            merged = []
            for b in sorted(set(final_beats)):
                if 0.10 <= b <= total_duration - 0.10 and (not merged or b - merged[-1] >= 0.10):
                    merged.append(b)
            final_beats = [0.0] + merged + [total_duration]

            self.set_status(f"{len(final_beats) - 2} Schnitte erfasst - bitte pruefen.", 0.0)
            self.open_manual_review(
                start_t, end_t, final_beats, raw_sorted, snapped, loose,
                master_audio_wav, onsets=detected_onsets, render_after=render_after
            )

        btn_row = ctk.CTkFrame(rec_win, fg_color="transparent")
        btn_row.pack(pady=(8, 14))
        btn_start = ctk.CTkButton(
            btn_row, text="Aufnahme starten", command=start_play,
            font=("Arial", 15, "bold"), height=42, width=220,
            fg_color="#b45309", hover_color="#92400e"
        )
        btn_start.pack(side="left", padx=6)
        ctk.CTkButton(
            btn_row, text="Fertig", command=finish, height=42, width=140,
            fg_color="#1f8a4c", hover_color="#176b3b", font=("Arial", 14, "bold")
        ).pack(side="left", padx=6)

        rec_win.bind("<Escape>", finish)
        rec_win.bind("<Return>", lambda e: (start_play(), "break")[1])
        rec_win.protocol("WM_DELETE_WINDOW", finish)
        self._center_window(rec_win, min_w=640, min_h=560)
        rec_win.after(80, lambda: (rec_win.grab_set(), rec_win.focus_force()))

    # =====================================================================
    # Main Generation Pipeline (Pro Studio Engine)
    # =====================================================================
    def generate_edit(self, start_t, end_t, out_path, custom_beat_times=None, mode="auto",
                      pre_master_audio=None, proxy=False, batch_label=None, settings=None):
        """
        Rendert einen Edit. Laeuft im Worker-Thread; liefert True bei Erfolg.
        Alle Einstellungen kommen als Snapshot (settings) vom UI-Thread.
        """
        temp_dir = tempfile.mkdtemp(prefix="autoedit_render_")
        temp_files = []
        success = False
        render_t0 = time.time()
        try:
            cfg = settings if settings else self.collect_render_settings()
            # Gespeicherte manuelle Cuts schlagen die automatische Erkennung -
            # sie sind bewusst gesetzt und werden fuer Vorschau, Einzelrender
            # und Batch-Export identisch verwendet.
            if custom_beat_times is None:
                custom_beat_times = cfg.get("manual_cuts") or None
            elif not custom_beat_times:
                custom_beat_times = None
            total_duration = end_t - start_t
            orientation = cfg["orientation"]
            scale_mode = cfg["scale_mode"]
            pace_choice = cfg["pace_choice"]
            fps = int(cfg["fps"])
            clip_audio_setting = cfg["clip_audio_setting"]
            enable_fade_out = cfg["enable_fade_out"]
            apply_cinematic_lut = cfg["apply_cinematic_lut"]
            enable_auto_loudness = cfg["enable_auto_loudness"]
            audio_source_path = cfg["audio_path"]
            video_paths = list(cfg["video_paths"])
            if not video_paths:
                raise ValueError("Keine Medien ausgewaehlt.")

            # Quality preset settings
            quality_str = cfg["quality_str"]
            if "Maximal" in quality_str:
                crf_val = "14"
                preset_val = "veryfast"
            elif "Ultra-Fast" in quality_str:
                crf_val = "20"
                preset_val = "ultrafast"
            else:
                crf_val = "16"
                preset_val = "veryfast"

            res_choice = cfg["res_choice"]

            # Effect controls & intensity
            intensity = cfg["fx_intensity"]
            allow_shake = cfg["allow_shake"]
            allow_bloom = cfg["allow_bloom"]
            allow_punch = cfg["allow_punch"]
            allow_vignette = cfg["allow_vignette"]
            allow_speedramp = cfg["allow_speedramp"]

            # Impact-Cadence: Akzent-Effekte auf jedem / jedem 2. / jedem 4. Cut
            cadence_step = 1
            if "Jeder 2. Cut" in cfg["fx_cadence"]:
                cadence_step = 2
            elif "Jeder 4. Cut" in cfg["fx_cadence"]:
                cadence_step = 4

            fx_cfg = get_effect_set(cfg["effect_set"])
            fx_cfg["cadence"] = cadence_step
            # Die Haekchen schalten einzelne Teilsysteme des Effekt-Sets ab
            if not allow_shake:
                fx_cfg["shake_on_accent"] = False
            if not allow_bloom:
                fx_cfg["flash_on_accent"] = False
            if not allow_punch:
                fx_cfg["punch"] = 0.0
                fx_cfg["whip_prob"] = 0.0
            if not cfg["fx_grain"]:
                fx_cfg["grain"] = 0.0
            if not allow_vignette:
                fx_cfg["vignette"] = False
            if not cfg["fx_zoom_blur"]:
                fx_cfg["zoom_blur_on_drop"] = False
                fx_cfg["freeze_on_drop"] = False
            if not cfg["fx_collage"]:
                fx_cfg["collage_prob"] = 0.0
            if not cfg["fx_spin"]:
                fx_cfg["spin_prob"] = 0.0
            allow_vignette = allow_vignette and bool(fx_cfg.get("vignette", True))

            grain_amount = float(fx_cfg.get("grain", 0.0))
            collage_prob = float(fx_cfg.get("collage_prob", 0.0))
            spin_prob = float(fx_cfg.get("spin_prob", 0.0))
            # Effekt-Placement ist reproduzierbar: gleiche Musik + gleicher Trim
            # => identisches Ergebnis. Batch-Varianten bekommen eigene Seeds.
            seed_src = (f"{os.path.basename(audio_source_path or '')}|{start_t:.3f}|"
                        f"{total_duration:.3f}|{cfg.get('variant', 0)}")
            render_seed = zlib.crc32(seed_src.encode("utf-8")) & 0x7FFFFFFF
            effect_rng = random.Random(render_seed)
            cut_rng = random.Random(render_seed + 101)

            # Jitter-Korrektur & Drop-Pre-Roll
            enable_snap = cfg["enable_snap"]
            preroll_choice = cfg["preroll_choice"]
            ai_audio_analysis = cfg["ai_audio_analysis"]
            overlay_manual_sec = cfg["overlay_manual_sec"]

            # KI-Auto-Reframe
            enable_auto_reframe = cfg["ai_reframe"] and orientation == "portrait"

            # Proxy-Modus (Schnell-Vorschau): NUR Aufloesung/FPS/CRF aendern.
            # Effekte bleiben identisch, sonst taugt die Vorschau nichts.
            if proxy:
                fps = min(30, int(cfg["fps"]) if cfg["fps"] else 30)
                crf_val = "30"
                preset_val = "ultrafast"
                res_choice = "Proxy (480p)"

            # Spinning Overlay Variables
            overlay_img = cfg["overlay_img_path"]
            has_overlay = bool(cfg["enable_overlay"] and overlay_img and os.path.exists(overlay_img))
            overlay_start_type = cfg["overlay_start_type"]
            overlay_corner_mode = cfg["overlay_corner_mode"]
            overlay_size_str = cfg["overlay_size_str"]
            overlay_speed_val = cfg["overlay_speed_val"]
            overlay_opacity_val = cfg["overlay_opacity_val"]
            overlay_circular = cfg["overlay_circular"]

            prefix = f"{batch_label} | " if batch_label else ""

            # Master audio extraction
            master_audio_path = os.path.join(temp_dir, "master_audio.wav")
            if pre_master_audio and os.path.exists(pre_master_audio):
                shutil.copy(pre_master_audio, master_audio_path)
            else:
                self.set_status(f"{prefix}Schritt 1/3: Lese Musik-Passage...", 0.04)
                run_ffmpeg([
                    self.ffmpeg_exe, "-y",
                    "-ss", f"{start_t:.3f}",
                    "-i", audio_source_path,
                    "-t", f"{total_duration:.3f}",
                    "-vn",
                    "-c:a", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    master_audio_path
                ])

            if self.is_cancelled:
                raise InterruptedError("Vorgang abgebrochen.")

            # -------------------------------------------------------------
            # STEP 1: Multi-Band Transient Audio-Analyse & Drop-Erkennung
            #
            # Laeuft auch bei manuellen Cuts: die Kick-Huelle entscheidet,
            # welche Cuts Akzent-Effekte (Punch, Flash, Shake) bekommen, und
            # der Drop bekommt den Zoom-Blur. Frueher wurde die Kick-Huelle
            # nur mit der (unsichtbaren) Bass-Pulse-Option berechnet - damit
            # feuerten Punch/Flash/Shake praktisch nie.
            # -------------------------------------------------------------
            self.set_status(f"{prefix}Schritt 1/3: Multi-Band Audio-Analyse & Transient-Tracking...", 0.08)
            sr = 22050
            y, _ = librosa.load(master_audio_path, sr=sr)
            y_percussive = librosa.effects.percussive(y, margin=2.2)
            env_low = librosa.onset.onset_strength(y=y_percussive, sr=sr, fmax=280)
            env_high = librosa.onset.onset_strength(y=y_percussive, sr=sr, fmin=800, fmax=5500)
            onset_env = env_low * 1.0 + env_high * 0.65

            kick_sr = sr
            kick_hop = 512
            kick_env = np.asarray(env_low, dtype=np.float32)
            kick_env = kick_env / (float(np.percentile(kick_env, 98)) + 1e-9)

            drop_timestamp = 0.0
            if ai_audio_analysis or (has_overlay and overlay_start_type == "drop"):
                try:
                    drop_timestamp = float(ai_detect_beat_drop(y, sr, total_duration))
                except Exception:
                    drop_timestamp = 0.0

            if self.is_cancelled:
                raise InterruptedError("Vorgang abgebrochen.")

            if custom_beat_times is None:
                if mode == "onsets":
                    self.set_status(f"{prefix}Schritt 1/3: Erfasse dynamische Onset-Peaks...", 0.12)
                    hop_len = 512
                    onset_frames = librosa.onset.onset_detect(
                        onset_envelope=onset_env, sr=sr, hop_length=hop_len,
                        backtrack=True, wait=6, delta=0.07
                    )
                    beat_times = [float(b) for b in librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_len)]
                    if enable_snap:
                        beat_times = snap_beats_to_transients(
                            beat_times, onset_env, sr, hop=hop_len, max_shift=0.06
                        )
                else:
                    self.set_status(f"{prefix}Schritt 1/3: Berechne Multi-Band Takt & Backtrack-Attack...", 0.12)
                    _tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                    beat_frames = librosa.onset.onset_backtrack(beat_frames, energy=onset_env)
                    raw_beats = librosa.frames_to_time(beat_frames, sr=sr)

                    rms = librosa.feature.rms(y=y)[0]
                    mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.01

                    # Drop auf den naechsten Beat legen
                    if drop_timestamp > 0.05 and len(raw_beats) > 0:
                        nearest_idx = int(np.argmin(np.abs(np.array(raw_beats) - drop_timestamp)))
                        drop_timestamp = float(raw_beats[nearest_idx])

                    beat_times = [float(b) for b in raw_beats]
                    # Leise Passagen: nur jeden zweiten Beat schneiden
                    if len(beat_times) > 1:
                        merged = []
                        idx = 0
                        while idx < len(beat_times) - 1:
                            t_cur = beat_times[idx]
                            t_next = beat_times[idx + 1]
                            f_start = librosa.time_to_frames(t_cur, sr=sr)
                            f_end = librosa.time_to_frames(t_next, sr=sr)
                            seg_rms = float(np.mean(rms[f_start:f_end])) if f_end > f_start else mean_rms
                            merged.append(t_cur)
                            if seg_rms < mean_rms * 0.60:
                                idx = min(len(beat_times) - 1, idx + 2)
                            else:
                                idx += 1
                        merged.append(beat_times[-1])
                        beat_times = merged

                    if enable_snap:
                        beat_times = snap_beats_to_transients(
                            beat_times, onset_env, sr, hop=512, max_shift=0.06
                        )

                beat_times = sorted(b for b in beat_times if 0.0 < b < total_duration)
                beat_times = [0.0] + beat_times + [total_duration]

                # Lange Luecken (> 2.5 s) gleichmaessig auffuellen
                dense_beats = [beat_times[0]]
                for k in range(len(beat_times) - 1):
                    t1, t2 = beat_times[k], beat_times[k + 1]
                    diff = t2 - t1
                    if diff > 2.5:
                        sub_cuts = int(np.ceil(diff / 1.5))
                        for s in range(1, sub_cuts):
                            dense_beats.append(t1 + (diff * s / sub_cuts))
                    dense_beats.append(t2)
                beat_times = dense_beats

                if "2x Schnell" in pace_choice:
                    fast_beats = []
                    for k in range(len(beat_times) - 1):
                        fast_beats.append(beat_times[k])
                        mid = (beat_times[k] + beat_times[k + 1]) / 2.0
                        if (mid - beat_times[k]) >= 0.15:
                            fast_beats.append(mid)
                    fast_beats.append(beat_times[-1])
                    beat_times = fast_beats
                elif "0.5x Entspannt" in pace_choice:
                    calm_beats = [beat_times[0]]
                    for k in range(1, len(beat_times) - 1, 2):
                        calm_beats.append(beat_times[k])
                    calm_beats.append(beat_times[-1])
                    beat_times = calm_beats

                sanitized_beats = [beat_times[0]]
                for b in beat_times[1:]:
                    if b - sanitized_beats[-1] >= 0.15:
                        sanitized_beats.append(b)
                sanitized_beats[-1] = total_duration
                beat_times = sanitized_beats
            else:
                self.set_status(f"{prefix}Schritt 1/3: Verwende {max(0, len(custom_beat_times) - 2)} manuelle Schnitte...", 0.15)
                inner = sorted(float(b) for b in custom_beat_times if 0.05 < float(b) < total_duration - 0.05)
                beat_times = [0.0] + inner + [total_duration]

            del y, y_percussive, env_low, env_high, onset_env

            if self.is_cancelled:
                raise InterruptedError("Vorgang abgebrochen.")

            # -------------------------------------------------------------
            # Drop-Pre-Roll: Edit startet N Beats vor dem Beat-Drop.
            # Der Audio-Start wird passend mitverschoben (edit_offset).
            # Nie bei manuellen Cuts - die sind bewusst so gesetzt.
            # -------------------------------------------------------------
            edit_offset = 0.0
            preroll_beats = 0
            for n_beats in (32, 16, 8, 4, 2):
                if preroll_choice.startswith(f"{n_beats} Beats"):
                    preroll_beats = n_beats
                    break

            if preroll_beats > 0 and drop_timestamp > 0.2 and custom_beat_times is None:
                inner = [beat_times[k + 1] - beat_times[k] for k in range(len(beat_times) - 1)]
                inner = [d for d in inner if 0.15 < d < 2.0]
                beat_len = float(np.median(inner)) if inner else 0.5
                target = drop_timestamp - (preroll_beats * beat_len)
                if target <= 0.3:
                    # Pre-Roll reicht ueber den Trim-Start hinaus: das komplette
                    # Intro bleibt drin.
                    edit_offset = 0.0
                elif target < total_duration - 1.0:
                    edit_offset = target
                    beat_times = [0.0] + [b - edit_offset for b in beat_times if b > edit_offset + 0.02]
                    drop_timestamp -= edit_offset
                    print(f"[Pre-Roll] {preroll_beats} Beats ({preroll_beats * beat_len:.2f}s) vor Drop, Offset {edit_offset:.2f}s")

            effective_duration = total_duration - edit_offset

            # Frame Quantization (Zero Accumulated Drift)
            raw_frame_indices = [int(round(t * fps)) for t in beat_times]
            total_target_frames = int(round(effective_duration * fps))
            min_cut_frames = 5 if fps >= 60 else 3
            clean_frame_indices = [0]
            for fi in raw_frame_indices[1:-1]:
                if fi - clean_frame_indices[-1] >= min_cut_frames and total_target_frames - fi >= min_cut_frames:
                    clean_frame_indices.append(fi)
            clean_frame_indices.append(total_target_frames)

            frame_counts = [clean_frame_indices[k + 1] - clean_frame_indices[k] for k in range(len(clean_frame_indices) - 1)]
            durations = [fc / float(fps) for fc in frame_counts]

            # Welcher Cut liegt auf dem Drop? Genau einer bekommt den Special-Look.
            active_drop_index = None
            if drop_timestamp > 0.05 and len(clean_frame_indices) > 1:
                starts = np.array(clean_frame_indices[:-1], dtype=np.float64) / float(fps)
                nearest = int(np.argmin(np.abs(starts - drop_timestamp)))
                if abs(starts[nearest] - drop_timestamp) < 0.35:
                    active_drop_index = clean_frame_indices[nearest]

            if has_overlay:
                if overlay_start_type == "start":
                    active_overlay_start = 0.0
                elif overlay_start_type == "drop":
                    active_overlay_start = drop_timestamp if drop_timestamp > 0 else max(0.0, overlay_manual_sec - edit_offset)
                else:
                    active_overlay_start = max(0.0, min(effective_duration, overlay_manual_sec - edit_offset))
            else:
                active_overlay_start = 999999.0

            # -------------------------------------------------------------
            # STEP 2: Video-Zuschnitt & Hardware-Beschleunigte Effekte
            # -------------------------------------------------------------
            self.set_status("Schritt 2/3: Berechne Canvas & Bild-Einpassung...", 0.20)

            orig_w, orig_h = 1920, 1080
            for fp in video_paths:
                f_ext = os.path.splitext(fp)[1].lower()
                try:
                    if f_ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                        with PIL.Image.open(fp) as _img:
                            orig_w, orig_h = _img.size
                    else:
                        _probe = VideoFileClip(fp)
                        orig_w, orig_h = _probe.size
                        _probe.close()
                    break
                except Exception:
                    continue

            # Resolution scaling
            if proxy:
                base_w, base_h = 480, 854
            elif "Auto" in res_choice:
                # Nie ueber die Quelle hinaus: hochskalierte Bilder sehen weich aus.
                base_w, base_h = max(2, orig_w), max(2, orig_h)
            elif "1440p" in res_choice:
                base_w, base_h = 2560, 1440
            elif "Original" in res_choice:
                base_w, base_h = orig_w, orig_h
            else:
                base_w, base_h = 1920, 1080

            if orientation == "portrait":
                tw, th = min(base_w, base_h), max(base_w, base_h)
            elif orientation == "landscape":
                tw, th = max(base_w, base_h), min(base_w, base_h)
            else:
                tw, th = orig_w, orig_h

            tw = tw - (tw % 2)
            th = th - (th % 2)

            video_positions = {}

            # Precompute Vignette Mask
            vignette_mask = _build_vignette_mask(tw, th) if allow_vignette else None

            # Pre-build Spinning Token
            prepared_token = None
            if "Groß" in overlay_size_str or "Gross" in overlay_size_str:
                token_ratio = 0.45
            elif "Mittel" in overlay_size_str:
                token_ratio = 0.30
            else:
                token_ratio = 0.20

            token_size = int(round(min(tw, th) * token_ratio))
            margin_x = int(round(tw * 0.035))
            margin_y = int(round(th * 0.035))

            center_pos = ((tw - token_size) // 2, (th - token_size) // 2)
            corners = [
                (margin_x, margin_y),                             # Top-Left
                (tw - token_size - margin_x, margin_y),           # Top-Right
                (tw - token_size - margin_x, th - token_size - margin_y), # Bottom-Right
                (margin_x, th - token_size - margin_y)            # Bottom-Left
            ]

            if has_overlay:
                try:
                    raw_ov = PIL.Image.open(overlay_img).convert("RGBA")
                    if overlay_circular:
                        min_dim = min(raw_ov.width, raw_ov.height)
                        left = (raw_ov.width - min_dim) // 2
                        top = (raw_ov.height - min_dim) // 2
                        sq = raw_ov.crop((left, top, left + min_dim, top + min_dim)).resize((token_size, token_size), PIL.Image.LANCZOS)
                        mask = PIL.Image.new("L", (token_size, token_size), 0)
                        d = PIL.ImageDraw.Draw(mask)
                        d.ellipse((3, 3, token_size - 3, token_size - 3), fill=255)
                        prepared_token = PIL.Image.new("RGBA", (token_size, token_size), (0, 0, 0, 0))
                        prepared_token.paste(sq, (0, 0), mask)
                        d_out = PIL.ImageDraw.Draw(prepared_token)
                        d_out.ellipse((3, 3, token_size - 3, token_size - 3), outline=(255, 255, 255, 220), width=3)
                    else:
                        prepared_token = raw_ov.resize((token_size, token_size), PIL.Image.LANCZOS)
                    raw_ov.close()

                    if overlay_opacity_val < 0.999 and prepared_token is not None:
                        r_ch, g_ch, b_ch, a_ch = prepared_token.split()
                        a_ch = a_ch.point(lambda p: int(p * overlay_opacity_val))
                        prepared_token = PIL.Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))
                except Exception as e:
                    print(f"Overlay preparation error: {e}")
                    prepared_token = None

            reframer = FaceReframer() if enable_auto_reframe else None

            def fit_clip(clip, target_w, target_h):
                """
                Bringt einen Clip auf die Zielgroesse - ohne Bildpunkte zu erfinden.

                Frueher wurde im Crop-Modus zuerst auf die Deckgroesse hochskaliert
                und danach zugeschnitten. Bei 720p-Quellen im Hochformat bedeutete
                das ein 1.8x-Upscale, das die Schaerfe sichtbar zerstoerte
                (Laplacian-Varianz der Quelle 283, im Render nur noch 80-175).
                Jetzt wird zuerst in Originalaufloesung der groesste moegliche
                Ausschnitt im Zielformat gesucht, danach genau einmal skaliert.

                Wichtig: Am Ende ist die Groesse garantiert exakt
                target_w x target_h. Die Effekt-Pipeline rechnet mit festen
                Masks (Vignette, LUT, Korn) in Zielgroesse - ein abweichendes
                Zwischenformat laesst cv2 mit mismatched sizes abstuerzen.
                """
                cw, ch = clip.size
                if cw <= 0 or ch <= 0:
                    return clip

                src_ratio = cw / float(ch)
                dst_ratio = target_w / float(target_h)

                if scale_mode == "fit":
                    # Schwarze Balken, aber niemals hochskaliert.
                    scale = min(target_w / float(cw), target_h / float(ch), 1.0)
                    resized = clip.resize(scale) if scale < 0.999 else clip
                    if tuple(resized.size) == (target_w, target_h):
                        return resized
                    return resized.on_color(
                        size=(target_w, target_h), color=(0, 0, 0), pos="center"
                    )

                # "fill": In Originalaufloesung so gross wie moeglich zuschneiden,
                # danach erst auf die Zielgroesse bringen.
                if src_ratio > dst_ratio:
                    # Quelle ist breiter -> seitlich zuschneiden, Hoehe bleibt
                    crop_w = min(int(round(ch * dst_ratio)), cw)
                    crop_h = ch
                else:
                    crop_w = cw
                    crop_h = min(int(round(cw / dst_ratio)), ch)

                left = (cw - crop_w) // 2
                top = (ch - crop_h) // 2
                cropped = clip.crop(
                    x1=left, y1=top, x2=left + crop_w, y2=top + crop_h
                )

                scale = min(target_w / float(crop_w), target_h / float(crop_h))
                # Hochskalieren nur, wenn das Ziel es wirklich verlangt (z.B. 1440p).
                # Ohne das bleiben 720p-Quellen bei 720p statt weich auf 1080p geblowen zu werden.
                resized = cropped.resize(scale) if abs(scale - 1.0) > 0.001 else cropped

                if tuple(resized.size) == (target_w, target_h):
                    base = resized
                else:
                    # Auf Zielgroesse bringen. Der Rest ist Rundungsrest vom
                    # Zielformat und wird zentriert weggeschnitten.
                    base = resized.resize((max(target_w, resized.size[0]),
                                                max(target_h, resized.size[1])))
                    lft = max(0, (base.size[0] - target_w) // 2)
                    ttop = max(0, (base.size[1] - target_h) // 2)
                    base = base.crop(x1=lft, y1=ttop, x2=lft + target_w, y2=ttop + target_h)

                if tuple(base.size) != (target_w, target_h):
                    # Letzter Sicherheitsnetz: exakt auf das Zielformat.
                    base = base.resize((target_w, target_h))

                if reframer is None:
                    return base

                # KI-Auto-Reframe: horizontal verschobener Crop, der dem Gesicht folgt
                def reframe_picker(gf, t):
                    frame = gf(t)
                    ratio = reframer.track_center(frame)
                    if ratio is None:
                        cx = base.size[0] / 2.0
                    else:
                        cx = ratio * base.size[0]
                        cx = max(target_w / 2.0, min(base.size[0] - target_w / 2.0, cx))
                    lft = int(round(cx - target_w / 2.0))
                    ttop = max(0, (base.size[1] - target_h) // 2)
                    lft = max(0, min(base.size[0] - target_w, lft))
                    return frame[ttop:ttop + target_h, lft:lft + target_w]

                return base.fl(reframe_picker)

            def open_source(path):
                ext_ = os.path.splitext(path)[1].lower()
                if ext_ in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                    if path in self.source_image_cache:
                        return ImageClip(self.source_image_cache[path])
                    img_ = PIL.Image.open(path).convert("RGB")
                    if img_.width > tw * 1.5 or img_.height > th * 1.5:
                        scale = max(tw / img_.width, th / img_.height)
                        nw, nh = max(1, int(img_.width * scale)), max(1, int(img_.height * scale))
                        img_ = img_.resize((nw, nh), resample=PIL.Image.BILINEAR)
                    arr_ = np.array(img_)
                    self.source_image_cache[path] = arr_
                    img_.close()
                    return ImageClip(arr_)
                else:
                    return VideoFileClip(path)

            def cut_source(src, dur, path=None):
                if isinstance(src, ImageClip):
                    return src.set_duration(dur)
                safe_max_dur = max(0.1, src.duration - 0.05)
                if path is not None and path in video_positions:
                    cs = video_positions[path]
                    if cs >= safe_max_dur:
                        cs = 0.0
                else:
                    max_s = max(0, safe_max_dur - dur)
                    cs = cut_rng.uniform(0, max_s) if max_s > 0 else 0.0
                end_s = min(cs + dur, safe_max_dur)
                sc = src.subclip(cs, end_s)
                if path is not None:
                    next_pos = end_s if end_s < safe_max_dur else 0.0
                    video_positions[path] = next_pos
                if sc.duration < dur:
                    sc = sc.fx(vfx.loop, duration=dur)
                return sc# =========================================================
            # Effekt-Engine (sparsam, CapCut-Stil)
            #
            # Grundregeln:
            #  - Die Mehrheit der Cuts bleibt sauber (kein Zoom, kein Shake)
            #  - Effekte sitzen auf dem Transient und klingen in <=200ms ab
            #  - Keine Dauerbewegung ueber den ganzen Cut
            #  - Jeder Effekt ist ein eigener, bewusst gesetzter Akzent
            # =========================================================
            def apply_pro_effects(clip, dur, cut_idx=0, cut_global_start=0.0, is_drop_cut=False):
                kick_strength = 0.0
                if kick_env is not None:
                    try:
                        # kick_env ist auf die Musik-Passage indiziert, die Cut-Zeiten
                        # nach dem Pre-Roll aber auf den Edit -> Offset addieren.
                        k_idx = int(round(((cut_global_start + edit_offset) * kick_sr) / kick_hop))
                        if 0 <= k_idx < len(kick_env):
                            k_hi = min(len(kick_env), k_idx + 8)
                            kick_strength = float(np.clip(np.max(kick_env[k_idx:k_hi]), 0.0, 1.0))
                    except Exception:
                        kick_strength = 0.0

                plan = build_effect_plan(
                    cut_idx, total_dur_count, cut_global_start, kick_strength,
                    is_drop_cut, fx_cfg, intensity, effect_rng
                )

                punch_amt = plan["amount"]
                punch_dur = max(0.01, plan["duration"])
                whip_amt = plan["amount"] if plan["transform"] == "whip_pan" else 0.0
                whip_dur = max(0.01, plan["duration"])
                whip_angle = plan["angle"]
                flash_amt = plan["flash"]
                shake_amt = plan["shake"]
                zoom_blur_amt = plan["zoom_blur"]

                has_warp = punch_amt > 0.0005 or whip_amt > 0.0005 or shake_amt > 0.05
                has_finish = (grain_amount > 0.0) or (allow_vignette and vignette_mask is not None) \
                    or apply_cinematic_lut or flash_amt > 0.0 or zoom_blur_amt > 0.0

                if not has_warp and not has_finish:
                    return clip

                need_margin = whip_amt > 0.0 or punch_amt > 0.02 or shake_amt > 0.05
                # Rand nur solange noetig, wie wirklich etwas transformiert wird
                margin_until = 0.0
                if need_margin:
                    margin_until = max(punch_dur if punch_amt > 0.02 else 0.0,
                                       whip_dur if whip_amt > 0.0 else 0.0,
                                       0.10 if shake_amt > 0.05 else 0.0)
                grain_field = build_grain_field(th, tw, grain_amount, render_seed + 13)
                grain_counter = [0]

                def process_frame(gf, t):
                    frame = gf(t)

                    # Whip-Pan und Punch-In brauchen Rand, sonst sieht man
                    # beim Verschieben die Bildkante.
                    if need_margin and t < margin_until:
                        margin = 1.0 + max(whip_amt * 1.6, punch_amt * 1.4, shake_amt / 60.0 + 0.02)
                        if margin > 1.005:
                            # Seitenverhaeltnis behalten - frueher wurde hier auf ein
                            # Quadrat skaliert, das Hochformat-Bild also waehrend jedes
                            # Punch/Whip sichtbar gestaucht.
                            mw = max(tw, int(round(tw * margin)))
                            mh = max(th, int(round(th * margin)))
                            frame = cv2.resize(frame, (mw, mh), interpolation=cv2.INTER_LINEAR)
                            xo = (mw - tw) // 2
                            yo = (mh - th) // 2
                            frame = frame[yo:yo + th, xo:xo + tw]
                            if frame.shape[0] != th or frame.shape[1] != tw:
                                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)

                    zoom = 1.0
                    dx = 0.0
                    dy = 0.0
                    ang = 0.0

                    # Punch-In / Punch-Out: Peak bei t=0, Ease-Out
                    if punch_amt > 0.0005 and t < punch_dur:
                        prog = max(0.0, 1.0 - t / punch_dur)
                        ease = prog * prog * (3.0 - 2.0 * prog)
                        if plan["transform"] == "punch_out":
                            zoom = 1.0 - (punch_amt * ease)
                        else:
                            zoom = 1.0 + (punch_amt * ease)

                    # Whip-Pan: linearer Versatz mit Bewegungsunschärfe
                    if whip_amt > 0.0 and t < whip_dur:
                        prog = max(0.0, 1.0 - t / whip_dur)
                        ease = prog * prog * (3.0 - 2.0 * prog)
                        sign = 1.0 if whip_angle == 0.0 else -1.0
                        dx = sign * whip_amt * tw * ease

                    # Kurzes, schnelles Shake auf starken Akzenten
                    if shake_amt > 0.05 and t < 0.10:
                        decay = (max(0.0, 1.0 - t / 0.10)) ** 2.2
                        dx += np.sin(t * 52.0) * shake_amt * decay
                        dy += np.cos(t * 44.0) * shake_amt * 0.7 * decay

                    if abs(zoom - 1.0) > 0.0008 or abs(dx) > 0.4 or abs(dy) > 0.4 or abs(ang) > 0.01:
                        M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), ang, zoom)
                        M[0, 2] += dx
                        M[1, 2] += dy
                        frame = cv2.warpAffine(
                            frame, M, (tw, th),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                        )

                    # Bewegungsunschärfe nur waehrend des Whip-Pans
                    if whip_amt > 0.0 and 0.0 <= t < whip_dur:
                        prog = max(0.0, 1.0 - t / whip_dur)
                        blur_len = int(round(3 + 9 * prog))
                        if blur_len >= 3:
                            frame = motion_blur_frame(frame, blur_len, whip_angle)

                    # Zoom-Blur-Burst auf dem Drop
                    if zoom_blur_amt > 0.0 and t < 0.22:
                        prog = max(0.0, 1.0 - t / 0.22)
                        frame = radial_zoom_blur(frame, zoom_blur_amt * prog)

                    # Flash: 2-3 Frames, Peak exakt auf dem Transient
                    if flash_amt > 0.0 and t < 0.05:
                        prog = max(0.0, 1.0 - t / 0.05)
                        gain = flash_amt * (prog ** 1.6)
                        if gain > 0.004:
                            frame = np.clip(
                                frame.astype(np.float32) * (1.0 + gain * 1.6)
                                + (255.0 * gain * 0.30),
                                0, 255
                            ).astype(np.uint8)

                    # --- Finishing: das, was fertig aussehen laesst ---
                    if allow_vignette and vignette_mask is not None:
                        frame = cv2.multiply(frame, vignette_mask, scale=1.0 / 255.0)

                    if apply_cinematic_lut:
                        frame = cv2.LUT(frame, CINEMATIC_LUT)

                    if grain_amount > 0.0:
                        frame = apply_film_grain(frame, grain_field, grain_counter[0])
                        grain_counter[0] += 1

                    return frame

                return clip.fl(process_frame)
            # Spinning Overlay Compositor
            def apply_spinning_overlay(clip, cur_cut_idx, cut_global_start, dur):
                if prepared_token is None:
                    return clip

                if "In der Mitte" in overlay_corner_mode:
                    target_pos = center_pos
                    prev_pos = center_pos
                elif "Wechselt zum Takt" in overlay_corner_mode:
                    target_corner_idx = cur_cut_idx % 4
                    prev_corner_idx = (cur_cut_idx - 1) % 4
                    target_pos = corners[target_corner_idx]
                    prev_pos = corners[prev_corner_idx]
                elif "Oben Links" in overlay_corner_mode:
                    target_pos = corners[0]
                    prev_pos = corners[0]
                elif "Oben Rechts" in overlay_corner_mode:
                    target_pos = corners[1]
                    prev_pos = corners[1]
                elif "Unten Rechts" in overlay_corner_mode:
                    target_pos = corners[2]
                    prev_pos = corners[2]
                else:
                    target_pos = corners[3]
                    prev_pos = corners[3]

                def overlay_frame(gf, t):
                    frame = gf(t)
                    global_t = cut_global_start + t

                    # Frame-accurate: only show overlay once trigger is reached
                    if global_t < active_overlay_start:
                        return frame

                    rel_t = global_t - active_overlay_start
                    slide_dur = 0.12
                    if "Wechselt zum Takt" in overlay_corner_mode and t < slide_dur and cur_cut_idx > 0:
                        prog = min(1.0, t / slide_dur)
                        ease = 1.0 - ((1.0 - prog) ** 3)
                        cx = int(prev_pos[0] + (target_pos[0] - prev_pos[0]) * ease)
                        cy = int(prev_pos[1] + (target_pos[1] - prev_pos[1]) * ease)
                    else:
                        cx, cy = target_pos

                    angle = (rel_t * 360.0 * overlay_speed_val) % 360.0
                    rotated = prepared_token.rotate(-angle, resample=PIL.Image.BILINEAR)

                    bg_pil = PIL.Image.fromarray(frame.astype(np.uint8))
                    bg_pil.paste(rotated, (cx, cy), rotated)
                    return np.array(bg_pil)

                return clip.fl(overlay_frame)

            if "Dezent 15%" in clip_audio_setting:
                clip_audio_vol = 0.15
            elif "Mittel 35%" in clip_audio_setting:
                clip_audio_vol = 0.35
            else:
                clip_audio_vol = 0.0

            clip_audio_sr = 44100
            clip_audio_buf = None
            if clip_audio_vol > 0.0:
                clip_audio_buf = np.zeros((int(np.ceil(effective_duration * clip_audio_sr)) + clip_audio_sr, 2),
                                          dtype=np.float32)

            def add_clip_audio(audio_clip, start_s, dur):
                """Clip-Ton an seiner Edit-Position in die Sammelspur addieren."""
                seg = audio_clip.subclip(0, min(dur, float(audio_clip.duration or dur)))
                arr = np.asarray(seg.to_soundarray(fps=clip_audio_sr, quantize=False), dtype=np.float32)
                if arr.ndim == 1:
                    arr = np.stack([arr, arr], axis=1)
                elif arr.shape[1] == 1:
                    arr = np.repeat(arr, 2, axis=1)
                elif arr.shape[1] > 2:
                    arr = arr[:, :2]
                i0 = int(round(start_s * clip_audio_sr))
                n = min(len(arr), len(clip_audio_buf) - i0)
                if n > 0:
                    clip_audio_buf[i0:i0 + n] += arr[:n] * clip_audio_vol

            cpu_threads = max(4, min(16, os.cpu_count() or 4))
            clip_idx = 0
            i = 0
            total_dur_count = len(durations)
            clip_loop_t0 = time.time()
            prefix = f"{batch_label} | " if batch_label else ""

            # -------------------------------------------------------------
            # Collage & Spin (aus dem Original-Repo uebernommen)
            # Die verbrauchten Beats werden uebersprungen, damit das
            # Beat-Gitter synchron bleibt.
            #  - 2x2 braucht mindestens 3 freie Beats, sonst fuellt das Raster
            #    nicht und man sieht statt eines Gitters ein einzelnes Bild.
            #  - das 4x4-Raster ist ein Highlight und landet nur kurz vor dem Drop.
            # -------------------------------------------------------------
            collage_seen = 0
            collage16_seen = 0
            drop_window_s = 8.0
            drop_start_t = (active_drop_index / float(fps)) if active_drop_index is not None else None

            def apply_finishing(clip):
                """Korn/Vignette/LUT - auch fuer Collage & Spin, sonst sehen
                die Raster-Segmente roh aus und fallen aus dem Bild."""
                grain_field = build_grain_field(th, tw, grain_amount, render_seed + 13)
                counter = [0]

                def finisher(gf, t):
                    frame = gf(t)
                    if allow_vignette and vignette_mask is not None:
                        frame = cv2.multiply(frame, vignette_mask, scale=1.0 / 255.0)
                    if apply_cinematic_lut:
                        frame = cv2.LUT(frame, CINEMATIC_LUT)
                    if grain_field is not None:
                        frame = apply_film_grain(frame, grain_field, counter[0])
                        counter[0] += 1
                    return frame

                if grain_field is None and not apply_cinematic_lut and not (allow_vignette and vignette_mask is not None):
                    return clip
                return clip.fl(finisher)

            def build_collage_4(start_idx, steps):
                """Hintergrund + bis zu 4 Kacheln, die Beat fuer Beat einblenden."""
                num_steps = min(4, steps)
                step_durs = [max(float(durations[start_idx + 1 + j]), 0.1) for j in range(num_steps)]
                total_dur = durations[start_idx] + sum(step_durs)

                bg = fit_clip(source_clip, tw, th).set_duration(total_dur)
                qw, qh = max(2, tw // 2), max(2, th // 2)
                quad_positions = [(0, 0), (tw // 2, 0), (0, th // 2), (tw // 2, th // 2)]
                layers = [bg]
                extra = []

                t_offset = durations[start_idx]
                for q in range(num_steps):
                    q_path = video_paths[(clip_idx + 1 + q) % len(video_paths)]
                    q_src = open_source(q_path)
                    extra.append(q_src)
                    q_rem = sum(step_durs[q:])
                    tile = fit_clip(cut_source(q_src, q_rem, q_path), qw, qh)
                    tile = tile.without_audio().set_start(t_offset).set_position(quad_positions[q])
                    layers.append(tile)
                    t_offset += step_durs[q]

                comp = CompositeVideoClip(layers, size=(tw, th)).without_audio().set_duration(total_dur)
                return comp, total_dur, num_steps, extra

            def build_collage_16(start_idx, steps):
                """4x4-Raster, 4 Kacheln pro Beat."""
                num_steps = min(4, max(1, steps // 4))
                step_durs = [max(float(durations[start_idx + 1 + j]), 0.1) for j in range(num_steps)]
                total_dur = durations[start_idx] + sum(step_durs)

                bg = fit_clip(source_clip, tw, th).set_duration(total_dur)
                gw, gh = max(2, tw // 4), max(2, th // 4)
                grid_positions = [(c * gw, r * gh) for r in range(4) for c in range(4)]
                layers = [bg]
                extra = []

                t_offset = durations[start_idx]
                tile_idx = 0
                for step in range(num_steps):
                    step_rem = sum(step_durs[step:])
                    for _ in range(4):
                        if tile_idx >= 16:
                            break
                        t_path = video_paths[(clip_idx + 1 + tile_idx) % len(video_paths)]
                        t_src = open_source(t_path)
                        extra.append(t_src)
                        tile = fit_clip(cut_source(t_src, step_rem, t_path), gw, gh)
                        tile = tile.without_audio().set_start(t_offset).set_position(grid_positions[tile_idx])
                        layers.append(tile)
                        tile_idx += 1
                    t_offset += step_durs[step]

                comp = CompositeVideoClip(layers, size=(tw, th)).without_audio().set_duration(total_dur)
                return comp, total_dur, num_steps, extra

            def build_spin(start_idx, steps):
                src = fit_clip(source_clip, tw, th)
                dur = durations[start_idx]

                def spin_frame(gf, t):
                    frame = gf(t)
                    angle = (t / max(0.01, dur)) * 360.0
                    img = PIL.Image.fromarray(frame)
                    return np.array(img.rotate(-angle, resample=PIL.Image.BILINEAR))

                spun = src.fl(spin_frame)
                big = spun.resize(1.42)
                return big.crop(
                    x_center=big.w / 2, y_center=big.h / 2, width=tw, height=th
                ).without_audio().set_duration(dur), dur, 0, []

            def write_segment(clip_to_write, seg_has_audio=False):
                temp_path = os.path.join(temp_dir, f"seg_{len(temp_files):04d}.mp4")
                write_args = {
                    "fps": fps,
                    "codec": "libx264",
                    "threads": cpu_threads,
                    "logger": None,
                    "ffmpeg_params": [
                        "-preset", preset_val,
                        "-crf", crf_val,
                        # "-tune film" glaettet Kontkanten sichtbar weg - genau die
                        # Art von Detailverlust, die man als "schlechte Qualitaet"
                        # einordnet, obwohl CRF und Aufloesung stimmen. Stattdessen
                        # aq=3 (adaptiver Quantisierer) fuer mehr Details in
                        # dunklen und stark bewegten Passagen.
                        "-profile:v", "high",
                        "-pix_fmt", "yuv420p",
                        "-x264-params", "aq-mode=3:aq-strength=1.0:deblock=-1,-1",
                        "-movflags", "+faststart",
                    ]
                }
                if seg_has_audio:
                    write_args["audio_codec"] = "aac"
                clip_to_write.write_videofile(temp_path, **write_args)
                temp_files.append(temp_path)
                try:
                    clip_to_write.close()
                except Exception:
                    pass

            while i < total_dur_count:
                if self.is_cancelled:
                    raise InterruptedError("Vorgang durch Benutzer abgebrochen.")

                duration = float(durations[i])
                if duration < 0.10:
                    duration = 0.10

                cut_global_start = clean_frame_indices[i] / float(fps)
                cur_path = video_paths[clip_idx]
                source_clip = open_source(cur_path)

                # Der Drop-Cut bekommt den Zoom-Blur-Burst und optional ein Freeze
                is_drop_cut = (active_drop_index is not None and clean_frame_indices[i] == active_drop_index)

                # Determine effect mode for this clip
                use_speedramp = allow_speedramp and not isinstance(source_clip, ImageClip) and (duration < 0.8) and (i % 3 == 0)
                if use_speedramp:
                    speed_mult = 1.30
                    raw_cut = cut_source(source_clip, duration * speed_mult, cur_path)
                    sub_clip = fit_clip(raw_cut, tw, th).fx(vfx.speedx, speed_mult)
                    if sub_clip.duration < duration:
                        sub_clip = sub_clip.fx(vfx.loop, duration=duration)
                    else:
                        sub_clip = sub_clip.subclip(0, duration)
                else:
                    raw_cut = cut_source(source_clip, duration, cur_path)
                    sub_clip = fit_clip(raw_cut, tw, th)

                # Freeze-Frame direkt auf dem Drop (2-3 Frames stehen bleiben)
                if is_drop_cut and fx_cfg.get("freeze_on_drop") and not isinstance(source_clip, ImageClip):
                    try:
                        freeze_t = min(2.5 / float(fps), duration * 0.35)
                        held = sub_clip.to_ImageClip(t=0, duration=freeze_t)
                        rest = sub_clip.subclip(freeze_t, duration)
                        sub_clip = concatenate_videoclips(
                            [held, rest if rest.duration > 0.01 else sub_clip.to_ImageClip(t=0, duration=duration - freeze_t)],
                            method="chain"
                        ).set_duration(duration)
                    except Exception:
                        pass

                # Effekte anwenden
                sub_clip = apply_pro_effects(
                    sub_clip, duration, cut_idx=i,
                    cut_global_start=cut_global_start, is_drop_cut=is_drop_cut
                )

                # Collage / Spin - verbrauchen mehrere Beats, das Gitter bleibt synchron
                available_after = total_dur_count - i - 1
                # Abstand bis zum Drop in Beats -> "kurz vor dem Drop"?
                near_drop = False
                if drop_start_t is not None:
                    gap = drop_start_t - cut_global_start
                    if 0.0 < gap <= drop_window_s:
                        near_drop = True

                do_16 = (
                    near_drop
                    and collage_prob > 0.0
                    and available_after >= 4
                    and collage16_seen < 1
                    and effect_rng.random() < 0.55
                )
                do_collage = (
                    not do_16
                    and collage_prob > 0.0
                    and available_after >= 3
                    and collage_seen < 5
                    and effect_rng.random() < collage_prob
                )
                do_spin = (
                    not do_16 and not do_collage and spin_prob > 0.0
                    and effect_rng.random() < spin_prob
                )

                if do_16 or do_collage or do_spin:
                    label = "Collage"
                    extra_sources = []
                    try:
                        if do_16:
                            seg_clip, seg_dur, consumed, extra_sources = build_collage_16(i, available_after)
                            label = "16-Collage"
                        elif do_collage:
                            seg_clip, seg_dur, consumed, extra_sources = build_collage_4(i, available_after)
                            label = "Collage"
                        else:
                            seg_clip, seg_dur, consumed, extra_sources = build_spin(i, available_after)
                            label = "Spin"

                        if has_overlay and (cut_global_start + seg_dur >= active_overlay_start):
                            seg_clip = apply_spinning_overlay(seg_clip, i, cut_global_start, seg_dur)

                        seg_clip = apply_finishing(seg_clip)
                        write_segment(seg_clip, False)
                        source_clip.close()
                        for e in extra_sources:
                            try:
                                e.close()
                            except Exception:
                                pass

                        collage_seen += 1
                        if label == "16-Collage":
                            collage16_seen += 1
                        clip_idx = (clip_idx + 1 + max(0, consumed)) % len(video_paths)
                        i += 1 + max(0, consumed)

                        pct = 0.20 + (0.68 * (i / total_dur_count))
                        self.set_status(
                            f"{prefix}Schritt 2/3: {label} ({len(temp_files)}/{total_dur_count})...", pct
                        )
                        continue
                    except Exception as e:
                        print(f"[{label}] fehlgeschlagen: {e}")
                        for e2 in extra_sources:
                            try:
                                e2.close()
                            except Exception:
                                pass
                        try:
                            source_clip.close()
                        except Exception:
                            pass
                        source_clip = open_source(cur_path)
                        raw_cut = cut_source(source_clip, duration, cur_path)
                        sub_clip = fit_clip(raw_cut, tw, th)
                        use_speedramp = False

                # Apply Spinning Overlay if clip overlaps or is past active_overlay_start
                if has_overlay and (cut_global_start + duration >= active_overlay_start):
                    sub_clip = apply_spinning_overlay(sub_clip, i, cut_global_start, duration)

                # Original-Clip-Sound: wird in einer eigenen Spur gesammelt und
                # beim finalen Mux unter die Musik gemischt. Frueher landete er
                # nur in den Segmenten und wurde beim Mux verworfen.
                if clip_audio_buf is not None:
                    try:
                        if raw_cut.audio is not None:
                            seg_audio = raw_cut.audio
                            if use_speedramp:
                                seg_audio = seg_audio.fx(vfx.speedx, speed_mult)
                            add_clip_audio(seg_audio, cut_global_start, duration)
                    except Exception as e:
                        print(f"[Clip-Audio] {e}")

                write_segment(sub_clip.without_audio())

                source_clip.close()

                clip_idx = (clip_idx + 1) % len(video_paths)
                i += 1

                pct = 0.20 + (0.68 * (i / total_dur_count))
                self.set_status(f"{prefix}Schritt 2/3: Erstelle Beat-Clip {len(temp_files)}/{total_dur_count} ({fps} FPS, {tw}x{th})...", pct)
                if i < total_dur_count:
                    elapsed = time.time() - clip_loop_t0
                    eta = (elapsed / i) * (total_dur_count - i)
                    self.set_eta(f"Restzeit: {format_eta(eta)}")
                else:
                    self.set_eta("")

            if self.is_cancelled:
                raise InterruptedError("Vorgang durch Benutzer abgebrochen.")
            if not temp_files:
                raise RuntimeError("Es wurde kein einziges Segment erzeugt.")

            # -------------------------------------------------------------
            # STEP 3: Concat + Studio Master Audio Mux
            # -------------------------------------------------------------
            self.set_status(f"{prefix}Schritt 3/3: Finale Zusammensetzung...", 0.90)

            concat_txt = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_txt, "w", encoding="utf-8") as f:
                for tp in temp_files:
                    f.write(f"file '{os.path.abspath(tp).replace(os.sep, '/')}'\n")

            visual_concat_mp4 = os.path.join(temp_dir, "visual_concat.mp4")
            try:
                run_ffmpeg([
                    self.ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_txt, "-c", "copy", visual_concat_mp4
                ])
            except RuntimeError:
                run_ffmpeg([
                    self.ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_txt, "-c:v", "libx264", "-preset", preset_val,
                    "-crf", crf_val, "-pix_fmt", "yuv420p", visual_concat_mp4
                ])

            self.set_status(f"{prefix}Schritt 3/3: Muxe Master-Audio & Fade-Out...", 0.95)

            master_chain = []
            if edit_offset > 0.01:
                master_chain.append(f"atrim=start={edit_offset:.3f},asetpts=PTS-STARTPTS")
            post_chain = []
            if enable_fade_out:
                post_chain.append(f"afade=t=out:st={max(0.0, effective_duration - 0.8):.3f}:d=0.8")
            if enable_auto_loudness:
                post_chain.append("loudnorm=I=-14:TP=-1.5:LRA=11")
            post_chain.append("aresample=48000")

            inputs = ["-i", visual_concat_mp4, "-i", master_audio_path]
            clip_wav = None
            if clip_audio_buf is not None and float(np.max(np.abs(clip_audio_buf))) > 1e-4:
                clip_wav = os.path.join(temp_dir, "clip_audio.wav")
                from scipy.io import wavfile
                pcm = np.clip(clip_audio_buf, -1.0, 1.0)
                wavfile.write(clip_wav, 44100, (pcm * 32767.0).astype(np.int16))
                inputs += ["-i", clip_wav]

            if clip_wav:
                m = ",".join(master_chain) if master_chain else "anull"
                graph = (f"[1:a]{m}[m];[m][2:a]amix=inputs=2:duration=first:normalize=0,"
                         + ",".join(post_chain) + "[aout]")
            else:
                graph = "[1:a]" + ",".join(master_chain + post_chain) + "[aout]"

            # Erst in eine Temp-Datei, dann verschieben: ein Abbruch oder Fehler
            # hinterlaesst keine halbe Datei am Zielort.
            final_tmp = os.path.join(temp_dir, "final_output.mp4")
            run_ffmpeg([
                self.ffmpeg_exe, "-y", *inputs,
                "-filter_complex", graph,
                "-map", "0:v:0", "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "320k",
                "-t", f"{effective_duration:.3f}",
                "-movflags", "+faststart",
                final_tmp
            ])
            try:
                out_dir = os.path.dirname(os.path.abspath(out_path))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                shutil.move(final_tmp, out_path)
            except PermissionError:
                raise RuntimeError(
                    f"Die Zieldatei ist gesperrt (vermutlich noch im Player geoeffnet):\n{out_path}"
                )

            self.last_rendered_path = out_path
            success = True
            self.set_eta("")
            elapsed_total = time.time() - render_t0
            if proxy:
                self.set_status(f"Vorschau fertig ({format_eta(elapsed_total)}) - wird abgespielt.", 1.0)
                self._ui_after(0, lambda: self._open_file(out_path))
            else:
                self.set_status(
                    f"{prefix}Fertig in {format_eta(elapsed_total)}: {os.path.basename(out_path)}", 1.0
                )
                if not batch_label:
                    self.show_render_done(out_path)
                self._ui_after(0, lambda: self.post_action_frame.pack(side="left", padx=(14, 0)))

        except InterruptedError:
            self.set_status("Rendern abgebrochen. Keine Datei erstellt.", 0.0)
        except Exception as e:
            traceback.print_exc()
            msg = str(e) or e.__class__.__name__
            self.set_status(f"Fehler: {msg.splitlines()[0][:140]}", 0.0)
            self.show_error("Fehler beim Rendern", f"{msg}\n\nDetails stehen in der Konsole.")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.set_eta("")
            self.render_finished.set()
            if not self.batch_running:
                self._ui_after(0, lambda: self._set_render_buttons(False))
        return success

    def _open_file(self, path):
        try:
            os.startfile(path)
        except Exception as e:
            self.show_error("Fehler", f"Datei konnte nicht geoeffnet werden:\n{e}")


if __name__ == "__main__":
    app = EditMakerApp()
    app.mainloop()
