import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
import librosa
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip, vfx, ImageClip
import os
import threading
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
import warnings
warnings.filterwarnings('ignore')

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

def ai_detect_beat_drop(y, sr, total_dur):
    """
    KI-basierte Struktur-Analyse & Beat-Drop Erkennung (Change-Point Detection).

    Kandidaten-Punkte kommen aus zwei Quellen:
      1. Agglomerative Segmentierung (Chroma-CQT + MFCC) -> musikalische Abschnitte
      2. Steile Energiesprünge im gleitenden Mehrband-Profil

    Bewertet wird jeder Kandidat als Energie-Sprung ueber die Grenze hinweg
    (RMS, Sub-Bass 30-160 Hz, Air 4-11 kHz). Der Drop ist die Grenze, die in den
    lautesten nachfolgenden Abschnitt fuehrt - nicht die mit dem groessten
    relativen Sprung. Beginnt der Track mit einer stillen Intro, gewinnt der
    erste echte Drop (Intro-Regel).
    """
    if total_dur < 3.5:
        return 0.0

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

        candidates = set()
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
            n_clusters = max(3, min(8, int(total_dur // 7)))
            bounds = librosa.segment.agglomerative(np.vstack([chroma, mfcc]), n_clusters)
            candidates.update(
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
            if b_r > a_r * 1.25 and b_s > a_s * 1.12:
                candidates.add(round(float(frames_t[i]), 2))

        candidates = sorted(c for c in candidates if 1.2 < c < total_dur * 0.90)
        if not candidates:
            return 0.0

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

        valid = []
        best = None
        for c in candidates:
            jump, after, before = _measure(c)
            if jump < 0.50:
                continue
            if after <= before * 1.05 or after < 0.45:
                continue
            valid.append(c)
            level = after - 0.35 * before
            if best is None or level > best[0]:
                best = (level, c)

        if best is None:
            return 0.0

        drop_t = best[1]

        if valid:
            first = valid[0]
            i_first = int(np.searchsorted(frames_t, first))
            pre_intro = _window_mean(p_rms, max(0, i_first - 2 * win), max(1, i_first))
            if pre_intro < 0.35:
                drop_t = first

        idx = int(np.searchsorted(frames_t, drop_t))
        lo, hi = max(0, idx - 6), min(n_frames, idx + 7)
        attack = 0.6 * p_sub[lo:hi] + 0.4 * p_rms[lo:hi]
        if len(attack) > 0:
            drop_t = float(frames_t[lo + int(np.argmax(attack))])

        return float(drop_t)
    except Exception as e:
        print(f"[AI Audio Error] {e}")
        return 0.0


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

# Default Presets
DEFAULT_PRESETS = {
    "TikTok Montage / Beat Drop": {
        "mode": "auto",
        "pace": "1x (Jeder Takt)",
        "orientation": "portrait",
        "scale_mode": "fill",
        "fps": 60,
        "quality": "High Quality (CRF 16 Studio)",
        "resolution": "1080p (Full HD)",
        "clip_audio": "Stumm (100% Musik)",
        "fade_out": True,
        "combo_style": "Impact-Combo (Gleichzeitig auf Beat-Hits gestapelt)",
        "fx_intensity": 130,
        "fx_shake": True,
        "fx_bloom_flash": True,
        "fx_punch_zoom": True,
        "fx_slow_push": True,
        "fx_rgb_shock": True,
        "fx_vignette": True,
        "fx_speedramp": True,
        "fx_cinematic_lut": True,
        "enable_overlay": False,
        "overlay_corner": "In der Mitte (Zentriert)",
        "overlay_size": "Gross (45% Bildhoehe)",
        "overlay_speed": 1.8,
        "overlay_opacity": 100,
        "overlay_start_mode": "drop",
        "overlay_circle": True,
        "fx_bass_pulse": False,
        "ai_reframe": False,
        "auto_loudness": True,
        "snap_jitter": True,
        "preroll": "8 Beats davor (2 Takte)",
        "fx_cadence": "Jeder 2. Cut (Sauberer Takt)"
        },
    "Velocity / Action Edit": {
        "mode": "onsets",
        "pace": "2x Schnell (Halber Takt)",
        "orientation": "portrait",
        "scale_mode": "fill",
        "fps": 60,
        "quality": "High Quality (CRF 16 Studio)",
        "resolution": "1080p (Full HD)",
        "clip_audio": "Dezent 15% (Killsounds/SFX)",
        "fade_out": True,
        "combo_style": "Impact-Combo (Gleichzeitig auf Beat-Hits gestapelt)",
        "fx_intensity": 100,
        "fx_shake": True,
        "fx_bloom_flash": True,
        "fx_punch_zoom": True,
        "fx_slow_push": False,
        "fx_rgb_shock": True,
        "fx_vignette": True,
        "fx_speedramp": True,
        "fx_cinematic_lut": True,
        "enable_overlay": False,
        "overlay_corner": "Wechselt zum Takt (Animiert)",
        "overlay_size": "Mittel (30% Bildhoehe)",
        "overlay_speed": 1.2,
        "overlay_opacity": 100,
        "overlay_start_mode": "drop",
        "overlay_circle": True,
        "fx_bass_pulse": False,
        "ai_reframe": False,
        "auto_loudness": True,
        "snap_jitter": True,
        "preroll": "4 Beats davor (1 Takt)",
        "fx_cadence": "Jeder Cut (Maximal)"
        },
    "Cinematic Flow & Aesthetic": {
        "mode": "auto",
        "pace": "1x (Jeder Takt)",
        "orientation": "landscape",
        "scale_mode": "fill",
        "fps": 60,
        "quality": "Maximal (CRF 14 Glasklar)",
        "resolution": "1440p (2K Ultra Sharp)",
        "clip_audio": "Stumm (100% Musik)",
        "fade_out": True,
        "combo_style": "Dezent & Sanft (Nur weiche Uebergaenge)",
        "fx_intensity": 70,
        "fx_shake": False,
        "fx_bloom_flash": True,
        "fx_punch_zoom": False,
        "fx_slow_push": True,
        "fx_rgb_shock": False,
        "fx_vignette": True,
        "fx_speedramp": False,
        "fx_cinematic_lut": True,
        "enable_overlay": False,
        "overlay_corner": "Oben Rechts",
        "overlay_size": "Kompakt (20% Bildhoehe)",
        "overlay_speed": 1.0,
        "overlay_opacity": 85,
        "overlay_start_mode": "start",
        "overlay_circle": True,
        "fx_bass_pulse": False,
        "ai_reframe": False,
        "auto_loudness": True,
        "snap_jitter": True,
        "preroll": "16 Beats davor (4 Takte)",
        "fx_cadence": "Jeder 2. Cut (Sauberer Takt)"
        },
    "Clean & Minimal (Pure Cuts)": {
        "mode": "auto",
        "pace": "1x (Jeder Takt)",
        "orientation": "portrait",
        "scale_mode": "fill",
        "fps": 60,
        "quality": "High Quality (CRF 16 Studio)",
        "resolution": "1080p (Full HD)",
        "clip_audio": "Stumm (100% Musik)",
        "fade_out": True,
        "combo_style": "Dezent & Sanft (Nur weiche Uebergaenge)",
        "fx_intensity": 40,
        "fx_shake": False,
        "fx_bloom_flash": False,
        "fx_punch_zoom": False,
        "fx_slow_push": False,
        "fx_rgb_shock": False,
        "fx_vignette": False,
        "fx_speedramp": False,
        "fx_cinematic_lut": False,
        "enable_overlay": False,
        "overlay_corner": "In der Mitte (Zentriert)",
        "overlay_size": "Mittel (30% Bildhoehe)",
        "overlay_speed": 1.0,
        "overlay_opacity": 100,
        "overlay_start_mode": "start",
        "overlay_circle": True,
        "fx_bass_pulse": False,
        "ai_reframe": False,
        "auto_loudness": True,
        "snap_jitter": True,
        "preroll": "Aus (Edit startet am Drop)",
        "fx_cadence": "Jeder 4. Cut (Nur Akzente)"
        }
}

# =====================================================================
# Theme & Configuration
# =====================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class EditMakerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Auto Beat-Sync Edit Maker (Pro Studio Edition)")
        self.geometry("820x970")
        self.minsize(780, 820)

        self.video_paths = []
        self.audio_path = None
        self.audio_duration = 0.0
        self.is_preview_playing = False
        self.manual_times = []
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

        pygame.mixer.init()

        self._build_ui()
        self._last_state = self.capture_full_state()
        self._register_history_traces()

    def _load_presets_from_disk(self):
        presets = dict(DEFAULT_PRESETS)
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        presets.update(data)
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

    # =====================================================================
    # UI Building
    # =====================================================================
    def _build_ui(self):
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=15, pady=12)

        # 1. Header
        lbl_title = ctk.CTkLabel(
            self.main_container,
            text="Auto Beat-Sync Edit Maker",
            font=("Arial", 26, "bold")
        )
        lbl_title.pack(pady=(4, 2))
        
        lbl_sub = ctk.CTkLabel(
            self.main_container,
            text="Pro Studio Engine | Multi-Band Transient Sync | 60 FPS | OpenCV Hardware-Beschleunigung",
            text_color="#38bdf8",
            font=("Arial", 12)
        )
        lbl_sub.pack(pady=(0, 10))

        # Profile & Preset Management Frame
        profile_frame = ctk.CTkFrame(self.main_container)
        profile_frame.pack(fill="x", pady=5)

        header_prof = ctk.CTkFrame(profile_frame, fg_color="transparent")
        header_prof.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(header_prof, text="Profile & Presets (Speichern & Laden)", font=("Arial", 14, "bold")).pack(side="left")

        row_prof = ctk.CTkFrame(profile_frame, fg_color="transparent")
        row_prof.pack(fill="x", padx=12, pady=(3, 8))

        ctk.CTkLabel(row_prof, text="Profil:").pack(side="left", padx=(0, 6))
        self.preset_names = list(self.all_presets.keys())
        self.profile_var = ctk.StringVar(value=self.preset_names[0] if self.preset_names else "Standard")
        self.combo_profile = ctk.CTkComboBox(
            row_prof,
            values=self.preset_names,
            variable=self.profile_var,
            command=self.on_profile_selected,
            width=230
        )
        self.combo_profile.pack(side="left", padx=4)

        self.btn_save_profile = ctk.CTkButton(
            row_prof,
            text="Profil speichern",
            command=self.save_current_as_profile,
            fg_color="#1f8a4c",
            hover_color="#176b3b",
            width=120
        )
        self.btn_save_profile.pack(side="left", padx=4)

        self.btn_load_json = ctk.CTkButton(
            row_prof,
            text="JSON laden",
            command=self.import_profile_json,
            fg_color="#2b719e",
            hover_color="#1f5375",
            width=90
        )
        self.btn_load_json.pack(side="left", padx=4)

        self.btn_export_json = ctk.CTkButton(
            row_prof,
            text="Export",
            command=self.export_profile_json,
            fg_color="#444444",
            hover_color="#555555",
            width=75
        )
        self.btn_export_json.pack(side="left", padx=4)

        self.btn_delete_prof = ctk.CTkButton(
            row_prof,
            text="Loeschen",
            command=self.delete_current_profile,
            fg_color="#8b1e1e",
            hover_color="#631414",
            width=75
        )
        self.btn_delete_prof.pack(side="left", padx=4)

        self.btn_undo = ctk.CTkButton(
            row_prof, text="Undo", command=self.undo_action,
            fg_color="#334155", hover_color="#475569", width=65
        )
        self.btn_undo.pack(side="right", padx=4)

        self.btn_redo = ctk.CTkButton(
            row_prof, text="Redo", command=self.redo_action,
            fg_color="#334155", hover_color="#475569", width=65
        )
        self.btn_redo.pack(side="right", padx=4)

        self.btn_load_project = ctk.CTkButton(
            row_prof, text="Projekt laden", command=self.load_project,
            fg_color="#1f6f8a", hover_color="#165266", width=115
        )
        self.btn_load_project.pack(side="right", padx=4)

        self.btn_save_project = ctk.CTkButton(
            row_prof, text="Projekt speichern", command=self.save_project,
            fg_color="#1f8a4c", hover_color="#176b3b", width=120
        )
        self.btn_save_project.pack(side="right", padx=4)

        # 2. Media Selection Frame
        media_frame = ctk.CTkFrame(self.main_container)
        media_frame.pack(fill="x", pady=5)

        header_media = ctk.CTkFrame(media_frame, fg_color="transparent")
        header_media.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(header_media, text="1. Medien (Videos & Bilder)", font=("Arial", 14, "bold")).pack(side="left")
        self.lbl_videos_count = ctk.CTkLabel(header_media, text="0 Dateien ausgewählt", text_color="gray")
        self.lbl_videos_count.pack(side="right")

        btn_row_media = ctk.CTkFrame(media_frame, fg_color="transparent")
        btn_row_media.pack(fill="x", padx=12, pady=4)

        self.btn_select_videos = ctk.CTkButton(
            btn_row_media,
            text="Clips / Bilder wählen",
            command=self.select_videos,
            width=180
        )
        self.btn_select_videos.pack(side="left", padx=(0, 10))

        self.btn_clear_videos = ctk.CTkButton(
            btn_row_media,
            text="Zurücksetzen",
            command=self.clear_videos,
            fg_color="#444444",
            hover_color="#555555",
            width=100
        )
        self.btn_clear_videos.pack(side="left", padx=5)

        self.shuffle_var = ctk.BooleanVar(value=True)
        self.chk_shuffle = ctk.CTkCheckBox(
            btn_row_media,
            text="Zufällige Reihenfolge (Shuffle)",
            variable=self.shuffle_var
        )
        self.chk_shuffle.pack(side="right", padx=5)

        self.media_listbox = ctk.CTkTextbox(media_frame, height=65, font=("Consolas", 11), activate_scrollbars=True)
        self.media_listbox.pack(fill="x", padx=12, pady=(4, 10))
        self.media_listbox.insert("1.0", "Noch keine Dateien gewählt. Unterstützt MP4, MOV, MKV, AVI, PNG, JPG, WEBP...")
        self.media_listbox.configure(state="disabled")

        # 3. Audio Selection & Trim Frame
        audio_frame = ctk.CTkFrame(self.main_container)
        audio_frame.pack(fill="x", pady=5)

        header_audio = ctk.CTkFrame(audio_frame, fg_color="transparent")
        header_audio.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(header_audio, text="2. Musik (Beat & Song)", font=("Arial", 14, "bold")).pack(side="left")
        self.lbl_audio_status = ctk.CTkLabel(header_audio, text="Kein Song gewählt", text_color="gray")
        self.lbl_audio_status.pack(side="right")

        btn_row_audio = ctk.CTkFrame(audio_frame, fg_color="transparent")
        btn_row_audio.pack(fill="x", padx=12, pady=4)

        self.btn_select_audio = ctk.CTkButton(
            btn_row_audio,
            text="Song auswählen",
            command=self.select_audio,
            width=180
        )
        self.btn_select_audio.pack(side="left", padx=(0, 10))

        self.btn_audio_preview = ctk.CTkButton(
            btn_row_audio,
            text="Vorhören",
            command=self.toggle_audio_preview,
            fg_color="#2b719e",
            hover_color="#1f5375",
            width=110,
            state="disabled"
        )
        self.btn_audio_preview.pack(side="left", padx=5)

        slider_frame = ctk.CTkFrame(audio_frame, fg_color="transparent")
        slider_frame.pack(fill="x", padx=12, pady=(4, 10))

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

        # Waveform with beat & drop markers
        waveform_wrap = ctk.CTkFrame(audio_frame, fg_color="transparent")
        waveform_wrap.pack(fill="x", padx=12, pady=(2, 4))

        self.lbl_waveform_hint = ctk.CTkLabel(
            waveform_wrap,
            text="Wellenform: Klick setzt den Startpunkt",
            font=("Arial", 10),
            text_color="#94a3b8"
        )
        self.lbl_waveform_hint.pack(anchor="w")

        self.wave_canvas = Canvas(
            waveform_wrap,
            height=70,
            bg="#0b1220",
            highlightthickness=1,
            highlightbackground="#1e293b",
            cursor="hand2"
        )
        self.wave_canvas.pack(fill="x", pady=(2, 0))
        self.wave_canvas.bind("<Configure>", lambda e: self.draw_waveform())
        self.wave_canvas.bind("<Button-1>", self.on_waveform_click)
        self.wave_env = None
        self.wave_hop = 0.0
        self.wave_beats = []
        self.wave_drop = 0.0

        # 4. Beat Detection & Pacing Frame (Multi-Band Engine)
        sync_frame = ctk.CTkFrame(self.main_container)
        sync_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(sync_frame, text="3. Beat-Sync & Multi-Band Timing", font=("Arial", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 3))

        mode_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        mode_row.pack(fill="x", padx=12, pady=3)

        self.mode_var = ctk.StringVar(value="auto")
        ctk.CTkRadioButton(mode_row, text="Auto Takt (Kick & Snare Transient-Sync)", variable=self.mode_var, value="auto").pack(side="left", padx=8)
        ctk.CTkRadioButton(mode_row, text="Dynamisch (Alle Onsets)", variable=self.mode_var, value="onsets").pack(side="left", padx=8)
        ctk.CTkRadioButton(mode_row, text="Manuell (Mit Takt-Magnet)", variable=self.mode_var, value="manual").pack(side="left", padx=8)

        pace_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        pace_row.pack(fill="x", padx=12, pady=(4, 10))

        ctk.CTkLabel(pace_row, text="Schnitt-Tempo:").pack(side="left", padx=(0, 8))
        self.pace_var = ctk.StringVar(value="1x (Jeder Takt)")
        self.combo_pace = ctk.CTkComboBox(
            pace_row,
            values=["1x (Jeder Takt)", "2x Schnell (Halber Takt)", "0.5x Entspannt (Doppel-Takt)"],
            variable=self.pace_var,
            width=210
        )
        self.combo_pace.pack(side="left", padx=5)

        self.ai_audio_analysis_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            pace_row,
            text="KI-Struktur- & Drum-Isolation (ML-Cluster)",
            variable=self.ai_audio_analysis_var
        ).pack(side="left", padx=(15, 5))

        self.snap_jitter_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            pace_row,
            text="Jitter-Korrektur (Beats auf Transients snappen)",
            variable=self.snap_jitter_var
        ).pack(side="left", padx=(10, 5))

        preroll_row = ctk.CTkFrame(sync_frame, fg_color="transparent")
        preroll_row.pack(fill="x", padx=12, pady=(2, 4))

        ctk.CTkLabel(preroll_row, text="Drop-Pre-Roll / Intro:").pack(side="left", padx=(0, 8))
        self.preroll_var = ctk.StringVar(value="16 Beats davor (4 Takte)")
        ctk.CTkOptionMenu(
            preroll_row,
            values=[
                "Aus (Edit startet am Drop)",
                "2 Beats davor",
                "4 Beats davor (1 Takt)",
                "8 Beats davor (2 Takte)",
                "16 Beats davor (4 Takte)",
                "32 Beats davor (8 Takte)"
            ],
            variable=self.preroll_var,
            width=250
        ).pack(side="left", padx=5)

        ctk.CTkLabel(preroll_row, text="Best-Moment-Länge:").pack(side="left", padx=(18, 8))
        self.best_moment_len_var = ctk.StringVar(value="30s")
        ctk.CTkOptionMenu(
            preroll_row,
            values=["15s", "30s", "60s", "90s"],
            variable=self.best_moment_len_var,
            width=70
        ).pack(side="left", padx=5)

        self.btn_best_moment = ctk.CTkButton(
            preroll_row, text="Besten Hook finden", width=150, height=26,
            fg_color="#7c3aed", hover_color="#5b21b6",
            command=self.find_best_moment_ui
        )
        self.btn_best_moment.pack(side="left", padx=(10, 0))

        # 5. Format, Skalierung & Qualität
        format_frame = ctk.CTkFrame(self.main_container)
        format_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(format_frame, text="4. Format, Auflösung, Qualität & FPS", font=("Arial", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 3))
        
        row_fmt = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_fmt.pack(fill="x", padx=12, pady=3)

        ctk.CTkLabel(row_fmt, text="Format:").pack(side="left", padx=(0, 8))
        self.orient_var = ctk.StringVar(value="original")
        ctk.CTkRadioButton(row_fmt, text="Originalformat (Kein Zoom)", variable=self.orient_var, value="original").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_fmt, text="Hochkant (9:16)", variable=self.orient_var, value="portrait").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_fmt, text="Quer (16:9)", variable=self.orient_var, value="landscape").pack(side="left", padx=6)

        row_scale_fps = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_scale_fps.pack(fill="x", padx=12, pady=3)

        ctk.CTkLabel(row_scale_fps, text="Einpassung:").pack(side="left", padx=(0, 6))
        self.scale_mode_var = ctk.StringVar(value="fit")
        ctk.CTkRadioButton(row_scale_fps, text="Ganzes Bild (Fit)", variable=self.scale_mode_var, value="fit").pack(side="left", padx=5)
        ctk.CTkRadioButton(row_scale_fps, text="Vollbild (Crop)", variable=self.scale_mode_var, value="fill").pack(side="left", padx=5)

        ctk.CTkLabel(row_scale_fps, text="Framerate:").pack(side="left", padx=(15, 6))
        self.fps_var = ctk.IntVar(value=60)
        ctk.CTkRadioButton(row_scale_fps, text="60 FPS (Flüssig)", variable=self.fps_var, value=60).pack(side="left", padx=5)
        ctk.CTkRadioButton(row_scale_fps, text="30 FPS", variable=self.fps_var, value=30).pack(side="left", padx=5)

        self.ai_reframe_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            row_scale_fps,
            text="KI-Auto-Reframe (Gesicht im 9:16-Crop halten)",
            variable=self.ai_reframe_var
        ).pack(side="left", padx=(15, 5))

        row_quality = ctk.CTkFrame(format_frame, fg_color="transparent")
        row_quality.pack(fill="x", padx=12, pady=(3, 10))

        ctk.CTkLabel(row_quality, text="Export-Qualität:").pack(side="left", padx=(0, 6))
        self.quality_var = ctk.StringVar(value="High Quality (CRF 16 Studio)")
        ctk.CTkComboBox(
            row_quality,
            values=[
                "High Quality (CRF 16 Studio)",
                "Maximal (CRF 14 Glasklar)",
                "Ultra-Fast (CRF 20 Schnell)"
            ],
            variable=self.quality_var,
            width=230
        ).pack(side="left", padx=5)

        ctk.CTkLabel(row_quality, text="Auflösung:").pack(side="left", padx=(15, 6))
        self.res_var = ctk.StringVar(value="1080p (Full HD)")
        ctk.CTkComboBox(
            row_quality,
            values=[
                "1080p (Full HD)",
                "1440p (2K Ultra Sharp)",
                "Originalgröße der Dateien"
            ],
            variable=self.res_var,
            width=180
        ).pack(side="left", padx=5)

        # 6. Audio-Mixing & Fade-Out
        audio_opt_frame = ctk.CTkFrame(self.main_container)
        audio_opt_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(audio_opt_frame, text="5. Audio-Mixing & Ausblenden", font=("Arial", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 3))
        
        row_mix = ctk.CTkFrame(audio_opt_frame, fg_color="transparent")
        row_mix.pack(fill="x", padx=12, pady=(3, 10))

        ctk.CTkLabel(row_mix, text="Original Clip-Sound:").pack(side="left", padx=(0, 8))
        self.clip_audio_var = ctk.StringVar(value="Stumm (100% Musik)")
        ctk.CTkComboBox(
            row_mix,
            values=["Stumm (100% Musik)", "Dezent 15% (Killsounds/SFX)", "Mittel 35% (Dialoge/SFX)"],
            variable=self.clip_audio_var,
            width=210
        ).pack(side="left", padx=5)

        self.fade_out_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row_mix,
            text="Sanfter Audio- & Video-Fade-Out am Ende (0.8s)",
            variable=self.fade_out_var
        ).pack(side="left", padx=(15, 5))

        self.auto_loudness_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row_mix,
            text="Auto-Loudness (EBU R128, -14 LUFS)",
            variable=self.auto_loudness_var
        ).pack(side="left", padx=(10, 5))

        # 7. Modern High-Impact Effects & Cinematic Grading Frame
        effects_frame = ctk.CTkFrame(self.main_container)
        effects_frame.pack(fill="x", pady=5)

        eff_header = ctk.CTkFrame(effects_frame, fg_color="transparent")
        eff_header.pack(fill="x", padx=12, pady=(10, 3))
        ctk.CTkLabel(eff_header, text="6. Professionelle Impact-Effekte & Cinematic Look", font=("Arial", 14, "bold")).pack(side="left")

        preset_row = ctk.CTkFrame(effects_frame, fg_color="transparent")
        preset_row.pack(fill="x", padx=12, pady=3)

        ctk.CTkLabel(preset_row, text="Effekt-Stil:").pack(side="left", padx=(0, 8))
        self.combo_fx_style = ctk.StringVar(value="Dynamischer Rhythmus-Flow (Abwechslungsreich)")
        ctk.CTkComboBox(
            preset_row,
            values=[
                "Dynamischer Rhythmus-Flow (Abwechslungsreich)",
                "Impact-Combo (Volle Power auf jeden Beat)",
                "Zufällig wechselnd (Ein Effekt pro Cut)",
                "Dezent & Sanft (Nur weiche Übergänge)"
            ],
            variable=self.combo_fx_style,
            width=330
        ).pack(side="left", padx=5)

        ctk.CTkLabel(preset_row, text="Impact-Rhythmus:").pack(side="left", padx=(10, 6))
        self.fx_cadence_var = ctk.StringVar(value="Jeder 2. Cut (Sauberer Takt)")
        ctk.CTkOptionMenu(
            preset_row,
            values=[
                "Jeder Cut (Maximal)",
                "Jeder 2. Cut (Sauberer Takt)",
                "Jeder 4. Cut (Nur Akzente)"
            ],
            variable=self.fx_cadence_var,
            width=185
        ).pack(side="left", padx=5)

        self.fx_cinematic_lut = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            preset_row,
            text="Cinematic S-Curve Contrast & Color Pop",
            variable=self.fx_cinematic_lut
        ).pack(side="left", padx=(15, 5))

        # Intensity Slider Row
        intensity_row = ctk.CTkFrame(effects_frame, fg_color="transparent")
        intensity_row.pack(fill="x", padx=12, pady=(4, 2))

        self.lbl_fx_intensity = ctk.CTkLabel(intensity_row, text="Effekt-Intensität: 100% (Recoil & Bloom)", width=280, anchor="w")
        self.lbl_fx_intensity.pack(side="left", padx=(0, 10))

        self.slider_fx_intensity = ctk.CTkSlider(
            intensity_row,
            from_=20, to=200,
            command=lambda v: (self.lbl_fx_intensity.configure(text=f"Effekt-Intensität: {int(v)}% (Recoil & Bloom)"), self.note_change())
        )
        self.slider_fx_intensity.set(100)
        self.slider_fx_intensity.pack(side="left", fill="x", expand=True, padx=5)

        eff_grid = ctk.CTkFrame(effects_frame, fg_color="transparent")
        eff_grid.pack(fill="x", padx=12, pady=(6, 10))

        self.fx_shake = ctk.BooleanVar(value=True)
        self.fx_bloom_flash = ctk.BooleanVar(value=True)
        self.fx_punch_zoom = ctk.BooleanVar(value=True)
        self.fx_slow_push = ctk.BooleanVar(value=True)
        self.fx_rgb_shock = ctk.BooleanVar(value=True)
        self.fx_vignette = ctk.BooleanVar(value=True)
        self.fx_speedramp = ctk.BooleanVar(value=True)
        self.fx_bass_pulse = ctk.BooleanVar(value=False)

        ctk.CTkCheckBox(eff_grid, text="Bass-Shake (Rotations-Tilt & Recoil)", variable=self.fx_shake).grid(row=0, column=0, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Exposure Bloom-Flash (Highlight-Burst)", variable=self.fx_bloom_flash).grid(row=0, column=1, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Punch-Zoom (Optischer Ease)", variable=self.fx_punch_zoom).grid(row=0, column=2, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Radial RGB-Aberration (Linsenzerlegung)", variable=self.fx_rgb_shock).grid(row=1, column=0, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Cinematic Vignette (Dunkler Randfokus)", variable=self.fx_vignette).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Slow Push-In (Sanfte Kamerafahrt)", variable=self.fx_slow_push).grid(row=1, column=2, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Speed-Ramp (Sync-Safe Velocity)", variable=self.fx_speedramp).grid(row=2, column=0, sticky="w", padx=8, pady=3)
        ctk.CTkCheckBox(eff_grid, text="Bass-Pulse (Full-Frame Kick-Darkening)", variable=self.fx_bass_pulse).grid(row=2, column=1, sticky="w", padx=8, pady=3)

        # 8. Spinning Overlay Frame
        overlay_frame = ctk.CTkFrame(self.main_container)
        overlay_frame.pack(fill="x", pady=5)

        overlay_header = ctk.CTkFrame(overlay_frame, fg_color="transparent")
        overlay_header.pack(fill="x", padx=12, pady=(10, 4))

        self.enable_overlay_var = ctk.BooleanVar(value=False)
        self.chk_overlay = ctk.CTkCheckBox(
            overlay_header,
            text="Spinning Bild-Overlay aktivieren",
            variable=self.enable_overlay_var,
            command=self._toggle_overlay_ui,
            font=("Arial", 14, "bold")
        )
        self.chk_overlay.pack(side="left")

        self.overlay_options_frame = ctk.CTkFrame(overlay_frame, fg_color="transparent")

        # Row 1: Image Selection
        row_ov1 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov1.pack(fill="x", padx=12, pady=4)

        self.btn_select_overlay = ctk.CTkButton(
            row_ov1,
            text="Overlay-Bild wählen (PNG/JPG)",
            command=self.select_overlay_image,
            width=220
        )
        self.btn_select_overlay.pack(side="left", padx=(0, 10))

        self.lbl_overlay_name = ctk.CTkLabel(row_ov1, text="Kein Bild ausgewählt", text_color="gray")
        self.lbl_overlay_name.pack(side="left")

        # Row 2: Timing / Start-Point
        row_ov2 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov2.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(row_ov2, text="Start-Zeitpunkt:").pack(side="left", padx=(0, 8))
        self.overlay_start_mode = ctk.StringVar(value="custom")
        ctk.CTkRadioButton(row_ov2, text="Ab Beat-Drop / Sekunde", variable=self.overlay_start_mode, value="custom").pack(side="left", padx=6)
        ctk.CTkRadioButton(row_ov2, text="Von Anfang an (0.0s)", variable=self.overlay_start_mode, value="start").pack(side="left", padx=6)

        self.btn_detect_drop = ctk.CTkButton(
            row_ov2, text="Auto-Drop finden", width=130, height=26,
            fg_color="#334155", hover_color="#475569",
            command=self.auto_detect_drop_ui
        )
        self.btn_detect_drop.pack(side="right", padx=(6, 0))

        self.row_ov_slider = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        self.row_ov_slider.pack(fill="x", padx=12, pady=2)
        self.lbl_overlay_sec = ctk.CTkLabel(self.row_ov_slider, text="Start bei: 29.0s", width=120, anchor="w")
        self.lbl_overlay_sec.pack(side="left")
        self.slider_overlay_sec = ctk.CTkSlider(
            self.row_ov_slider, from_=0, to=120,
            command=lambda v: (self.lbl_overlay_sec.configure(text=f"Start bei: {v:.1f}s"), self.note_change())
        )
        self.slider_overlay_sec.set(29.0)
        self.slider_overlay_sec.pack(side="right", fill="x", expand=True, padx=5)

        # Row 3: Placement & Size
        row_ov3 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov3.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(row_ov3, text="Platzierung:").pack(side="left", padx=(0, 6))
        self.overlay_corner_var = ctk.StringVar(value="Wechselt zum Takt (Animiert)")
        self.combo_corner = ctk.CTkComboBox(
            row_ov3,
            values=[
                "Wechselt zum Takt (Animiert)",
                "In der Mitte (Zentriert)",
                "Oben Links",
                "Oben Rechts",
                "Unten Links",
                "Unten Rechts"
            ],
            variable=self.overlay_corner_var,
            width=220
        )
        self.combo_corner.pack(side="left", padx=5)

        ctk.CTkLabel(row_ov3, text="Größe:").pack(side="left", padx=(12, 6))
        self.overlay_size_var = ctk.StringVar(value="Mittel (30% - Standard)")
        self.combo_ov_size = ctk.CTkComboBox(
            row_ov3,
            values=[
                "Kompakt (20% - Dezent)",
                "Mittel (30% - Standard)",
                "Groß (45% - Fett & Dominant)"
            ],
            variable=self.overlay_size_var,
            width=210
        )
        self.combo_ov_size.pack(side="left", padx=5)

        self.overlay_circle_cut_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row_ov3,
            text="Kreis-Zuschnitt mit Rand",
            variable=self.overlay_circle_cut_var
        ).pack(side="left", padx=(12, 5))

        # Row 4: Speed & Opacity Slider
        row_ov4 = ctk.CTkFrame(self.overlay_options_frame, fg_color="transparent")
        row_ov4.pack(fill="x", padx=12, pady=(4, 12))

        speed_col = ctk.CTkFrame(row_ov4, fg_color="transparent")
        speed_col.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.lbl_ov_speed = ctk.CTkLabel(speed_col, text="Dreh-Speed: 1.2x (Umdr./s)", anchor="w")
        self.lbl_ov_speed.pack(fill="x")
        self.slider_ov_speed = ctk.CTkSlider(
            speed_col, from_=0.2, to=4.0,
            command=lambda v: (self.lbl_ov_speed.configure(text=f"Dreh-Speed: {v:.1f}x (Umdr./s)"), self.note_change())
        )
        self.slider_ov_speed.set(1.2)
        self.slider_ov_speed.pack(fill="x", pady=2)

        opacity_col = ctk.CTkFrame(row_ov4, fg_color="transparent")
        opacity_col.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self.lbl_ov_opacity = ctk.CTkLabel(opacity_col, text="Deckkraft: 100%", anchor="w")
        self.lbl_ov_opacity.pack(fill="x")
        self.slider_ov_opacity = ctk.CTkSlider(
            opacity_col, from_=10, to=100,
            command=lambda v: (self.lbl_ov_opacity.configure(text=f"Deckkraft: {int(v)}%"), self.note_change())
        )
        self.slider_ov_opacity.set(100.0)
        self.slider_ov_opacity.pack(fill="x", pady=2)

        # 9. Render & Controls Frame
        bottom_frame = ctk.CTkFrame(self.main_container)
        bottom_frame.pack(fill="x", pady=8)

        btn_row_action = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        btn_row_action.pack(fill="x", padx=20, pady=(12, 6))

        self.btn_generate = ctk.CTkButton(
            btn_row_action,
            text="Video Rendern (Pro Studio Engine)",
            command=self.start_generation,
            fg_color="#1f8a4c",
            hover_color="#176b3b",
            height=44,
            font=("Arial", 16, "bold")
        )
        self.btn_generate.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_proxy = ctk.CTkButton(
            btn_row_action,
            text="Schnell-Vorschau (480p)",
            command=self.start_proxy_render,
            fg_color="#7c3aed",
            hover_color="#5b21b6",
            height=44,
            width=190,
            font=("Arial", 13, "bold")
        )
        self.btn_proxy.pack(side="left", padx=(0, 10))

        self.btn_batch = ctk.CTkButton(
            btn_row_action,
            text="Batch-Export",
            command=self.start_batch_export,
            fg_color="#b45309",
            hover_color="#92400e",
            height=44,
            width=150,
            font=("Arial", 13, "bold")
        )
        self.btn_batch.pack(side="left", padx=(0, 10))

        self.btn_cancel = ctk.CTkButton(
            btn_row_action,
            text="Abbrechen",
            command=self.cancel_generation,
            fg_color="#8b1e1e",
            hover_color="#631414",
            height=44,
            width=110,
            font=("Arial", 14, "bold"),
            state="disabled"
        )
        self.btn_cancel.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(bottom_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=(4, 6))
        self.progress_bar.set(0.0)

        status_row = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        status_row.pack(fill="x", padx=20, pady=(0, 6))

        self.lbl_status = ctk.CTkLabel(status_row, text="Bereit für Schnitt", font=("Arial", 12), text_color="yellow")
        self.lbl_status.pack(side="left")

        self.lbl_pct = ctk.CTkLabel(status_row, text="0%", font=("Arial", 12, "bold"))
        self.lbl_pct.pack(side="right")

        self.lbl_eta = ctk.CTkLabel(status_row, text="", font=("Arial", 11), text_color="#94a3b8")
        self.lbl_eta.pack(side="right", padx=(0, 18))

        self.post_action_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        self.btn_open_folder = ctk.CTkButton(
            self.post_action_frame,
            text="Ordner im Explorer öffnen",
            command=self.open_output_folder,
            fg_color="#2b719e",
            hover_color="#1f5375",
            width=200
        )
        self.btn_open_folder.pack(side="left", padx=5)

        self.btn_play_video = ctk.CTkButton(
            self.post_action_frame,
            text="Video abspielen",
            command=self.play_output_video,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            width=150
        )
        self.btn_play_video.pack(side="left", padx=5)

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
                    self.lbl_overlay_sec.configure(text=f"Start bei: {clamped_t:.1f}s")
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

    def get_current_settings_dict(self):
        return {
            "mode": self.mode_var.get(),
            "pace": self.pace_var.get(),
            "orientation": self.orient_var.get(),
            "scale_mode": self.scale_mode_var.get(),
            "fps": int(self.fps_var.get()),
            "quality": self.quality_var.get(),
            "resolution": self.res_var.get(),
            "clip_audio": self.clip_audio_var.get(),
            "fade_out": self.fade_out_var.get(),
            "combo_style": self.combo_fx_style.get(),
            "fx_intensity": int(self.slider_fx_intensity.get()),
            "fx_shake": self.fx_shake.get(),
            "fx_bloom_flash": self.fx_bloom_flash.get(),
            "fx_punch_zoom": self.fx_punch_zoom.get(),
            "fx_slow_push": self.fx_slow_push.get(),
            "fx_rgb_shock": self.fx_rgb_shock.get(),
            "fx_vignette": self.fx_vignette.get(),
            "fx_speedramp": self.fx_speedramp.get(),
            "fx_cinematic_lut": self.fx_cinematic_lut.get(),
            "enable_overlay": self.enable_overlay_var.get(),
            "overlay_corner": self.overlay_corner_var.get(),
            "overlay_size": self.overlay_size_var.get(),
            "overlay_speed": float(self.slider_ov_speed.get()),
            "overlay_opacity": float(self.slider_ov_opacity.get()),
            "overlay_start_mode": self.overlay_start_mode.get(),
            "overlay_circle": self.overlay_circle_cut_var.get(),
            "fx_bass_pulse": self.fx_bass_pulse.get(),
            "ai_reframe": self.ai_reframe_var.get(),
            "auto_loudness": self.auto_loudness_var.get(),
            "snap_jitter": self.snap_jitter_var.get(),
            "preroll": self.preroll_var.get()
        }

    def apply_preset_dict(self, data):
        if not isinstance(data, dict):
            return
        if "mode" in data: self.mode_var.set(data["mode"])
        if "pace" in data: self.pace_var.set(data["pace"])
        if "orientation" in data: self.orient_var.set(data["orientation"])
        if "scale_mode" in data: self.scale_mode_var.set(data["scale_mode"])
        if "fps" in data: self.fps_var.set(int(data["fps"]))
        if "quality" in data: self.quality_var.set(data["quality"])
        if "resolution" in data: self.res_var.set(data["resolution"])
        if "clip_audio" in data: self.clip_audio_var.set(data["clip_audio"])
        if "fade_out" in data: self.fade_out_var.set(bool(data["fade_out"]))
        if "combo_style" in data: self.combo_fx_style.set(data["combo_style"])
        if "fx_intensity" in data:
            val = float(data["fx_intensity"])
            self.slider_fx_intensity.set(val)
            self.lbl_fx_intensity.configure(text=f"Effekt-Intensität: {int(val)}% (Recoil & Bloom)")
        if "fx_shake" in data: self.fx_shake.set(bool(data["fx_shake"]))
        if "fx_bloom_flash" in data: self.fx_bloom_flash.set(bool(data["fx_bloom_flash"]))
        if "fx_punch_zoom" in data: self.fx_punch_zoom.set(bool(data["fx_punch_zoom"]))
        if "fx_slow_push" in data: self.fx_slow_push.set(bool(data["fx_slow_push"]))
        if "fx_rgb_shock" in data: self.fx_rgb_shock.set(bool(data["fx_rgb_shock"]))
        if "fx_vignette" in data: self.fx_vignette.set(bool(data["fx_vignette"]))
        if "fx_speedramp" in data: self.fx_speedramp.set(bool(data["fx_speedramp"]))
        if "fx_cinematic_lut" in data: self.fx_cinematic_lut.set(bool(data["fx_cinematic_lut"]))
        if "enable_overlay" in data:
            self.enable_overlay_var.set(bool(data["enable_overlay"]))
            self._toggle_overlay_ui()
        if "overlay_corner" in data: self.overlay_corner_var.set(data["overlay_corner"])
        if "overlay_size" in data: self.overlay_size_var.set(data["overlay_size"])
        if "overlay_speed" in data:
            spd = float(data["overlay_speed"])
            self.slider_ov_speed.set(spd)
            self.lbl_ov_speed.configure(text=f"Dreh-Speed: {spd:.1f}x (Umdr./s)")
        if "overlay_opacity" in data:
            op = float(data["overlay_opacity"])
            self.slider_ov_opacity.set(op)
            self.lbl_ov_opacity.configure(text=f"Deckkraft: {int(op)}%")
        if "overlay_start_mode" in data: self.overlay_start_mode.set(data["overlay_start_mode"])
        if "overlay_circle" in data: self.overlay_circle_cut_var.set(bool(data["overlay_circle"]))
        if "fx_bass_pulse" in data: self.fx_bass_pulse.set(bool(data["fx_bass_pulse"]))
        if "fx_cadence" in data: self.fx_cadence_var.set(data["fx_cadence"])
        if "ai_reframe" in data: self.ai_reframe_var.set(bool(data["ai_reframe"]))
        if "auto_loudness" in data: self.auto_loudness_var.set(bool(data["auto_loudness"]))
        if "snap_jitter" in data: self.snap_jitter_var.set(bool(data["snap_jitter"]))
        if "preroll" in data: self.preroll_var.set(data["preroll"])

    def save_current_as_profile(self):
        dialog = ctk.CTkInputDialog(title="Profil speichern", text="Name für das neue Profil:")
        name = dialog.get_input()
        if name and name.strip():
            name = name.strip()
            self.all_presets[name] = self.get_current_settings_dict()
            self._save_presets_to_disk()
            values = list(self.all_presets.keys())
            self.combo_profile.configure(values=values)
            self.profile_var.set(name)
            self.set_status(f"Profil '{name}' erfolgreich gespeichert.", 0.0)

    def import_profile_json(self):
        path = filedialog.askopenfilename(
            title="Profil aus JSON-Datei laden",
            filetypes=[("JSON-Dateien", "*.json"), ("Alle Dateien", "*.*")]
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "mode" in data or "fps" in data:
                        base_name = os.path.splitext(os.path.basename(path))[0]
                        self.all_presets[base_name] = data
                        self._save_presets_to_disk()
                        self.combo_profile.configure(values=list(self.all_presets.keys()))
                        self.profile_var.set(base_name)
                        self.apply_preset_dict(data)
                        self.set_status(f"Profil aus '{os.path.basename(path)}' geladen.", 0.0)
                    else:
                        self.all_presets.update(data)
                        self._save_presets_to_disk()
                        self.combo_profile.configure(values=list(self.all_presets.keys()))
                        self.set_status("Preset-Bundle importiert.", 0.0)
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
                self.set_status(f"Profil nach '{os.path.basename(path)}' exportiert.", 0.0)
            except Exception as e:
                self.show_error("Fehler", f"Export fehlgeschlagen:\n{e}")

    def delete_current_profile(self):
        cur = self.profile_var.get()
        if cur in DEFAULT_PRESETS:
            messagebox.showinfo("Hinweis", f"Das Standard-Profil '{cur}' kann nicht gelöscht werden.")
            return
        if cur in self.all_presets:
            del self.all_presets[cur]
            self._save_presets_to_disk()
            values = list(self.all_presets.keys())
            self.combo_profile.configure(values=values)
            new_sel = values[0] if values else "Standard"
            self.profile_var.set(new_sel)
            if new_sel in self.all_presets:
                self.apply_preset_dict(self.all_presets[new_sel])
            self.set_status(f"Profil '{cur}' gelöscht.", 0.0)

    # =====================================================================
    # Undo / Redo (Settings- & Medien-Historie)
    # =====================================================================
    def _register_history_traces(self):
        tracked = [
            self.mode_var, self.pace_var, self.orient_var, self.scale_mode_var,
            self.fps_var, self.quality_var, self.res_var, self.clip_audio_var,
            self.fade_out_var, self.combo_fx_style, self.fx_cinematic_lut,
            self.fx_shake, self.fx_bloom_flash, self.fx_punch_zoom,
            self.fx_slow_push, self.fx_rgb_shock, self.fx_vignette,
            self.fx_speedramp, self.fx_bass_pulse, self.fx_cadence_var, self.ai_reframe_var,
            self.auto_loudness_var, self.snap_jitter_var, self.preroll_var,
            self.enable_overlay_var, self.overlay_corner_var,
            self.overlay_size_var, self.overlay_start_mode,
            self.overlay_circle_cut_var, self.shuffle_var,
            self.ai_audio_analysis_var, self.best_moment_len_var
        ]
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
                self.load_audio_into_ui(audio)
            if self.audio_duration > 0:
                self.slider_start.set(state.get("start", 0.0) or 0.0)
                self.slider_end.set(state.get("end", self.audio_duration) or self.audio_duration)
                self.lbl_start.configure(text=f"Start: {self.slider_start.get():.1f}s")
                self.lbl_end.configure(text=f"Ende: {self.slider_end.get():.1f}s")

            sec = float(state.get("overlay_sec", 0.0))
            if hasattr(self, 'slider_overlay_sec'):
                self.slider_overlay_sec.set(sec)
                self.lbl_overlay_sec.configure(text=f"Start bei: {sec:.1f}s")
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
        path = filedialog.asksaveasfilename(
            title="Projekt speichern",
            defaultextension=".aep",
            filetypes=[("AutoEdit Projekt", "*.aep *.json"), ("JSON-Dateien", "*.json")]
        )
        if not path:
            return
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
                "version": 2,
                "settings": self.get_current_settings_dict(),
                "videos": [_rel(p) for p in self.video_paths],
                "audio": _rel(self.audio_path),
                "overlay_img": _rel(self.overlay_img_path),
                "start": float(self.slider_start.get()) if self.audio_duration else 0.0,
                "end": float(self.slider_end.get()) if self.audio_duration else 0.0,
                "overlay_sec": float(self.slider_overlay_sec.get()),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.set_status(f"Projekt gespeichert: {os.path.basename(path)}", 0.0)
        except Exception as e:
            self.show_error("Fehler", f"Projekt konnte nicht gespeichert werden:\n{e}")

    def load_project(self):
        path = filedialog.askopenfilename(
            title="Projekt laden",
            filetypes=[("AutoEdit Projekt", "*.aep *.json"), ("JSON-Dateien", "*.json")]
        )
        if not path:
            return
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

            self.apply_full_state({
                "settings": data.get("settings", {}),
                "video_paths": videos,
                "audio_path": _abs(data.get("audio")),
                "overlay_img": _abs(data.get("overlay_img")),
                "overlay_sec": data.get("overlay_sec", 0.0),
                "start": data.get("start", 0.0),
                "end": data.get("end", 0.0),
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
        try:
            return self.after(delay_ms, func)
        except Exception:
            try:
                func()
            except Exception:
                pass
            return None

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
            win.geometry("560x290")
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
                btn_row, text="Ordner oeffnen", command=_open_folder,
                fg_color="#2b719e", hover_color="#1f5375", width=170, height=40
            ).pack(side="left", padx=8)

            ctk.CTkButton(
                btn_row, text="Schliessen", command=_close,
                fg_color="#444444", hover_color="#555555", width=120, height=40
            ).pack(side="left", padx=8)

            win.protocol("WM_DELETE_WINDOW", _close)
            win.after(250, lambda: win.attributes("-topmost", False))
            return win

        self._ui_after(0, _build)

    # =====================================================================
    # Media Selection & Preview
    # =====================================================================
    def select_videos(self):
        paths = filedialog.askopenfilenames(
            title="Wähle Videos oder Bilder aus",
            filetypes=[
                ("Medien-Dateien", "*.mp4 *.mov *.avi *.mkv *.png *.jpg *.jpeg *.webp *.bmp"),
                ("Video-Dateien", "*.mp4 *.mov *.avi *.mkv"),
                ("Bilder", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("Alle Dateien", "*.*")
            ]
        )
        if paths:
            new_paths = [p for p in paths if p not in self.video_paths]
            self.video_paths.extend(new_paths)
            self._update_media_listbox()
            self.note_change()

    def clear_videos(self):
        self.video_paths = []
        self._update_media_listbox()
        self.note_change()

    def _update_media_listbox(self):
        self.media_listbox.configure(state="normal")
        self.media_listbox.delete("1.0", "end")
        count = len(self.video_paths)
        if count == 0:
            self.media_listbox.insert("1.0", "Noch keine Dateien gewählt. Unterstützt MP4, MOV, MKV, AVI, PNG, JPG, WEBP...")
            self.lbl_videos_count.configure(text="0 Dateien ausgewählt", text_color="gray")
        else:
            names = [f"[{i+1}] {os.path.basename(p)}" for i, p in enumerate(self.video_paths)]
            self.media_listbox.insert("1.0", "\n".join(names))
            self.lbl_videos_count.configure(text=f"{count} Dateien bereit", text_color="#38bdf8")
        self.media_listbox.configure(state="disabled")

    def select_audio(self):
        path = filedialog.askopenfilename(
            title="Wähle eine Musik-Datei",
            filetypes=[("Audio-Dateien", "*.mp3 *.wav *.ogg *.aac *.flac *.m4a"), ("Alle Dateien", "*.*")]
        )
        if path:
            self.load_audio_into_ui(path)

    def load_audio_into_ui(self, path):
        if not path or not os.path.exists(path):
            return False
        self.audio_path = path
        try:
            probe = AudioFileClip(path)
            self.audio_duration = float(probe.duration)
            probe.close()
        except Exception:
            try:
                self.audio_duration = float(librosa.get_duration(path=path))
            except Exception as e:
                self.show_error("Fehler", f"Konnte Audio-Dauer nicht ermitteln:\n{e}")
                self.audio_path = None
                return False

        self.slider_start.configure(to=self.audio_duration)
        self.slider_end.configure(to=self.audio_duration)
        self.slider_start.set(0)
        self.slider_end.set(self.audio_duration)
        self.lbl_start.configure(text="Start: 0.0s")
        self.lbl_end.configure(text=f"Ende: {self.audio_duration:.1f}s")
        if hasattr(self, 'slider_overlay_sec'):
            self.slider_overlay_sec.configure(to=self.audio_duration)
        self.lbl_audio_status.configure(
            text=f"{os.path.basename(path)} ({self.audio_duration:.1f}s)",
            text_color="#38bdf8"
        )
        self.btn_audio_preview.configure(state="normal")

        self.wave_env = None
        self.wave_beats = []
        self.wave_drop = 0.0
        self.draw_waveform()
        self.analyze_waveform_async()
        return True

    def update_start_lbl(self, val):
        val = float(val)
        if val >= self.slider_end.get():
            val = max(0, self.slider_end.get() - 0.5)
            self.slider_start.set(val)
        self.lbl_start.configure(text=f"Start: {val:.1f}s")
        self.draw_waveform()
        self.note_change()

    def update_end_lbl(self, val):
        val = float(val)
        if val <= self.slider_start.get():
            val = min(self.audio_duration, self.slider_start.get() + 0.5)
            self.slider_end.set(val)
        self.lbl_end.configure(text=f"Ende: {val:.1f}s")
        self.draw_waveform()
        self.note_change()

    # =====================================================================
    # Waveform View (Beat- & Drop-Marker)
    # =====================================================================
    def analyze_waveform_async(self):
        if not self.audio_path or self.wave_analysis_running:
            return
        self.wave_analysis_running = True
        path = self.audio_path

        def _worker():
            try:
                env, hop, beats, drop = compute_audio_overview(path)
            except Exception as e:
                print(f"[Waveform Error] {e}")
                env, hop, beats, drop = None, 0.0, [], 0.0

            def _apply():
                self.wave_analysis_running = False
                if path != self.audio_path:
                    return
                self.wave_env = env
                self.wave_hop = hop
                self.wave_beats = beats
                self.wave_drop = drop
                self.draw_waveform()
                if drop > 0 and hasattr(self, 'slider_overlay_sec'):
                    self.lbl_waveform_hint.configure(
                        text=f"Wellenform: Klick setzt den Startpunkt | KI-Drop bei {drop:.1f}s",
                        text_color="#f97316"
                    )
                else:
                    self.lbl_waveform_hint.configure(
                        text="Wellenform: Klick setzt den Startpunkt",
                        text_color="#94a3b8"
                    )

            self._ui_after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def draw_waveform(self):
        if not hasattr(self, "wave_canvas"):
            return
        canvas = self.wave_canvas
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 2 or h <= 2:
            return

        dur = self.audio_duration
        if not dur or self.wave_env is None or len(self.wave_env) == 0:
            canvas.create_text(
                w // 2, h // 2, text="Wellenform wird analysiert...",
                fill="#475569", font=("Arial", 10)
            )
            return

        env = self.wave_env
        hop = max(1e-4, self.wave_hop)
        mid = h / 2.0
        max_v = float(np.percentile(env, 98)) or 1.0

        for b in self.wave_beats:
            x = int(b / dur * w)
            if 0 <= x < w:
                canvas.create_line(x, 0, x, h, fill="#16243b")

        top_pts = []
        bot_pts = []
        for x in range(0, w, 1):
            i0 = int(x / float(w) * dur / hop)
            i1 = int((x + 1) / float(w) * dur / hop)
            i0 = max(0, min(len(env) - 1, i0))
            i1 = max(i0 + 1, min(len(env), i1))
            seg = env[i0:i1]
            v = float(np.max(seg)) / max_v if len(seg) else 0.0
            amp = min(1.0, v) * (mid - 2)
            top_pts.extend((x, mid - amp))
            bot_pts.extend((x, mid + amp))

        if top_pts:
            polygon = top_pts + list(reversed(bot_pts))
            canvas.create_polygon(polygon, fill="#38bdf8", outline="")

        xs = int(self.slider_start.get() / dur * w)
        xe = int(self.slider_end.get() / dur * w)
        if xs > 0:
            canvas.create_rectangle(0, 0, xs, h, fill="#020617", stipple="gray50", outline="")
        if xe < w:
            canvas.create_rectangle(xe, 0, w, h, fill="#020617", stipple="gray50", outline="")

        canvas.create_line(xs, 0, xs, h, fill="#22c55e", width=2)
        canvas.create_line(xe, 0, xe, h, fill="#f97316", width=2)

        if self.wave_drop > 0:
            xd = int(self.wave_drop / dur * w)
            if 0 <= xd < w:
                canvas.create_line(xd, 0, xd, h, fill="#ef4444", width=2)
                label_x = xd + 4 if xd < w - 46 else xd - 4
                canvas.create_text(
                    label_x, 8, text="DROP", fill="#ef4444",
                    font=("Arial", 8, "bold"), anchor="w" if label_x > xd else "e"
                )

    def on_waveform_click(self, event):
        if not self.audio_duration or not self.wave_canvas.winfo_width():
            return
        frac = max(0.0, min(1.0, event.x / float(self.wave_canvas.winfo_width())))
        t = frac * self.audio_duration
        self.slider_start.set(t)
        self.slider_end.set(min(self.audio_duration, max(t + 0.5, self.slider_end.get())))
        self.lbl_start.configure(text=f"Start: {t:.1f}s")
        self.draw_waveform()

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
                self.slider_start.set(start)
                self.slider_end.set(end)
                self.lbl_start.configure(text=f"Start: {start:.1f}s")
                self.lbl_end.configure(text=f"Ende: {end:.1f}s")
                self.draw_waveform()
                if drop > 0 and self.slider_overlay_sec.get() > end:
                    self.slider_overlay_sec.set(max(0.0, min(start, drop - start)))
                    self.lbl_overlay_sec.configure(
                        text=f"Start bei: {max(0.0, min(start, drop - start)):.1f}s"
                    )
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
            self.btn_audio_preview.configure(text="Pause", fg_color="#c026d3", hover_color="#9333ea")

            def check_finished():
                if self.is_preview_playing and not pygame.mixer.music.get_busy():
                    self.stop_audio_preview()
                elif self.is_preview_playing:
                    self.after(100, check_finished)

            self.after(100, check_finished)
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
    def start_generation(self):
        if not self.video_paths:
            self.show_error("Fehler", "Bitte wähle mindestens ein Video oder Bild aus.")
            return
        if not self.audio_path:
            self.show_error("Fehler", "Bitte wähle eine Musik-Datei aus.")
            return

        start_t = self.slider_start.get()
        end_t = self.slider_end.get()
        if end_t <= start_t:
            self.show_error("Fehler", "Das Ende der Musik muss nach dem Start liegen.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Video speichern als",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not out_path:
            return

        self.is_cancelled = False
        self._set_render_buttons(True)
        self.post_action_frame.pack_forget()
        self.set_eta("")
        self.render_finished.clear()

        if self.shuffle_var.get():
            random.shuffle(self.video_paths)

        mode = self.mode_var.get()
        if mode == "manual":
            self.open_manual_recorder(start_t, end_t, out_path)
        else:
            self.set_status("Starte Beat-Analyse & Vorbereitung...", 0.05)
            render_settings = self.collect_render_settings()
            threading.Thread(
                target=self.generate_edit,
                args=(start_t, end_t, out_path, None, mode, None, False, None, render_settings),
                daemon=True
            ).start()

    def _collect_render_inputs(self):
        if not self.video_paths:
            self.show_error("Fehler", "Bitte wähle mindestens ein Video oder Bild aus.")
            return None
        if not self.audio_path:
            self.show_error("Fehler", "Bitte wähle eine Musik-Datei aus.")
            return None
        start_t = self.slider_start.get()
        end_t = self.slider_end.get()
        if end_t <= start_t:
            self.show_error("Fehler", "Das Ende der Musik muss nach dem Start liegen.")
            return None
        return start_t, end_t

    def collect_render_settings(self):
        """
        Nimmt alle Render-Optionen als einfachen Snapshot auf dem UI-Thread auf.
        So liest der Render-Worker keine Tk-Variablen aus einem Fremd-Thread.
        """
        return {
            "audio_path": self.audio_path,
            "video_paths": list(self.video_paths),
            "overlay_img_path": self.overlay_img_path,
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
            "fx_style": self.combo_fx_style.get(),
            "fx_intensity": float(self.slider_fx_intensity.get()) / 100.0,
            "allow_shake": self.fx_shake.get(),
            "allow_bloom": self.fx_bloom_flash.get(),
            "allow_punch": self.fx_punch_zoom.get(),
            "allow_push": self.fx_slow_push.get(),
            "allow_rgb": self.fx_rgb_shock.get(),
            "allow_vignette": self.fx_vignette.get(),
            "allow_speedramp": self.fx_speedramp.get(),
            "allow_bass_pulse": self.fx_bass_pulse.get(),
            "fx_cadence": self.fx_cadence_var.get(),
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
        for btn in (self.btn_generate, self.btn_proxy, self.btn_batch):
            btn.configure(state=state)
        self.btn_cancel.configure(state="normal" if busy else "disabled")

    def start_proxy_render(self):
        inputs = self._collect_render_inputs()
        if not inputs:
            return
        start_t, end_t = inputs

        suggested = os.path.join(
            os.path.dirname(self.audio_path) if self.audio_path else os.getcwd(),
            f"preview_{int(time.time())}.mp4"
        )
        out_path = filedialog.asksaveasfilename(
            title="Schnell-Vorschau speichern als (480p Proxy)",
            initialfile=os.path.basename(suggested),
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not out_path:
            return

        mode = self.mode_var.get()
        if mode == "manual":
            self.show_info("Hinweis", "Im manuellen Modus gibt es keine Schnell-Vorschau. Bitte den normalen Render nutzen.")
            return

        self.is_cancelled = False
        self.post_action_frame.pack_forget()
        self.set_eta("")
        self.render_finished.clear()
        self._set_render_buttons(True)
        self.set_status("Schnell-Vorschau: Analyse...", 0.05)
        render_settings = self.collect_render_settings()
        threading.Thread(
            target=self.generate_edit,
            args=(start_t, end_t, out_path, None, mode, None, True, None, render_settings),
            daemon=True
        ).start()

    def start_batch_export(self):
        inputs = self._collect_render_inputs()
        if not inputs:
            return
        start_t, end_t = inputs

        mode = self.mode_var.get()
        if mode == "manual":
            self.show_info("Hinweis", "Batch-Export ist nur im Auto- oder Onset-Modus verfügbar.")
            return

        dialog = ctk.CTkInputDialog(
            title="Batch-Export",
            text="Wie viele Varianten rendern? (jede mit gemischter Clip-Reihenfolge)"
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

        folder = filedialog.askdirectory(title="Zielordner für Batch-Export wählen")
        if not folder:
            return

        base_name = os.path.splitext(os.path.basename(self.video_paths[0]))[0]
        self.is_cancelled = False
        self.post_action_frame.pack_forget()
        self.set_eta("")
        self._set_render_buttons(True)
        render_settings = self.collect_render_settings()

        def _worker():
            done = 0
            last_out = None
            for i in range(count):
                if self.is_cancelled:
                    break
                shuffled = list(render_settings["video_paths"])
                random.shuffle(shuffled)
                variant_settings = dict(render_settings)
                variant_settings["video_paths"] = shuffled
                out_path = os.path.join(folder, f"{base_name}_batch_{i + 1:02d}.mp4")
                self.render_finished.clear()
                self.generate_edit(
                    start_t, end_t, out_path, None, mode, None, False,
                    batch_label=f"Batch {i + 1}/{count}",
                    settings=variant_settings
                )
                if os.path.exists(out_path):
                    done += 1
                    last_out = out_path
            self.set_eta("")
            if self.is_cancelled:
                self.set_status(f"Batch abgebrochen ({done}/{count} fertig).", 0.0)
            else:
                self.set_status(f"Batch-Export fertig: {done}/{count} Videos in {folder}", 1.0)
                if last_out and os.path.exists(last_out):
                    self.show_render_done(
                        last_out, extra_msg=f"{done} Videos im Ordner:\n{folder}"
                    )
                else:
                    self.show_info("Fertig", f"{done} Videos wurden exportiert nach:\n{folder}")
            self._ui_after(0, lambda: self._set_render_buttons(False))

        threading.Thread(target=_worker, daemon=True).start()

    # =====================================================================
    # Manual Recording Mode (Takt-Magnet)
    # =====================================================================
    def open_manual_recorder(self, start_t, end_t, out_path):
        total_duration = end_t - start_t
        temp_dir = tempfile.mkdtemp(prefix="autoedit_manual_")
        master_audio_wav = os.path.join(temp_dir, "master_audio.wav")
        audio_source_path = self.audio_path

        try:
            cmd = [
                self.ffmpeg_exe, "-y",
                "-ss", f"{start_t:.3f}",
                "-i", audio_source_path,
                "-t", f"{total_duration:.3f}",
                "-vn",
                "-c:a", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                master_audio_wav
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError("FFmpeg Audio-Extraktion fehlgeschlagen.")
        except Exception as e:
            self.show_error("Fehler", f"Audio-Vorbereitung fehlgeschlagen:\n{e}")
            self.btn_generate.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        detected_onsets = []
        try:
            y_m, sr_m = librosa.load(master_audio_wav, sr=22050)
            y_perc = librosa.effects.percussive(y_m, margin=2.0)
            env_low = librosa.onset.onset_strength(y=y_perc, sr=sr_m, fmax=280)
            env_high = librosa.onset.onset_strength(y=y_perc, sr=sr_m, fmin=800, fmax=5500)
            onset_env = env_low * 1.0 + env_high * 0.65
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_m, backtrack=True, wait=6, delta=0.07)
            detected_onsets = list(librosa.frames_to_time(onset_frames, sr=sr_m))
            del y_m, y_perc, env_low, env_high, onset_env
        except Exception:
            detected_onsets = []

        self.manual_times = []
        rec_win = ctk.CTkToplevel(self)
        rec_win.title("Manuelles Beat-Recording (Mit Takt-Magnet)")
        rec_win.geometry("520x330")
        rec_win.transient(self)
        rec_win.grab_set()

        lbl_info = ctk.CTkLabel(
            rec_win,
            text="Tippe im Takt der Musik: LEERTASTE, Pfeiltasten oder Maus!\nFenster schließen, wenn fertig.",
            font=("Arial", 14)
        )
        lbl_info.pack(pady=12)

        magnet_var = ctk.BooleanVar(value=True)
        chk_magnet = ctk.CTkCheckBox(
            rec_win,
            text="Takt-Magnet aktiv (Klicks an Drum-Hits einrasten)",
            variable=magnet_var
        )
        chk_magnet.pack(pady=(0, 8))

        lbl_cuts = ctk.CTkLabel(rec_win, text="Aufgenommene Beats: 0", font=("Arial", 18, "bold"), text_color="yellow")
        lbl_cuts.pack(pady=3)

        lbl_time = ctk.CTkLabel(rec_win, text="Zeit: 0.0s / 0.0s", font=("Arial", 12), text_color="gray")
        lbl_time.pack(pady=2)

        try:
            pygame.mixer.music.load(master_audio_wav)
        except Exception as e:
            self.show_error("Fehler", f"Pygame Audio-Ladefehler:\n{e}")
            rec_win.destroy()
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.btn_generate.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
            return

        is_playing = False
        start_perf_time = 0.0

        def record_cut(event=None):
            nonlocal is_playing, start_perf_time
            if not is_playing:
                return
            raw_time = time.perf_counter() - start_perf_time
            compensated_time = max(0.01, raw_time - 0.07)
            if 0.05 < compensated_time < total_duration:
                self.manual_times.append(compensated_time)
                lbl_cuts.configure(text=f"Aufgenommene Beats: {len(self.manual_times)}")

        for key in ("<space>", "<Left>", "<Right>", "<Up>", "<Down>"):
            rec_win.bind(key, record_cut)
        rec_win.bind("<Button-1>", record_cut)
        rec_win.bind("<Button-3>", record_cut)

        def update_playback_clock():
            if is_playing:
                elapsed = time.perf_counter() - start_perf_time
                lbl_time.configure(text=f"Zeit: {min(elapsed, total_duration):.1f}s / {total_duration:.1f}s")
                if elapsed >= total_duration:
                    on_close()
                    return
                rec_win.after(40, update_playback_clock)

        def start_play():
            nonlocal is_playing, start_perf_time
            is_playing = True
            btn_start.configure(state="disabled")
            start_perf_time = time.perf_counter()
            pygame.mixer.music.play()
            rec_win.focus_set()
            update_playback_clock()

        btn_start = ctk.CTkButton(rec_win, text="Aufnahme Starten", command=start_play, font=("Arial", 15, "bold"), height=40)
        btn_start.pack(pady=10)

        def on_close():
            nonlocal is_playing
            is_playing = False
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass
            rec_win.destroy()

            final_beats = []
            if len(self.manual_times) > 0:
                raw_sorted = sorted(list(set(self.manual_times)))
                if magnet_var.get() and len(detected_onsets) > 0:
                    onsets_arr = np.array(detected_onsets)
                    for tap in raw_sorted:
                        diffs = np.abs(onsets_arr - tap)
                        nearest_idx = int(np.argmin(diffs))
                        if diffs[nearest_idx] <= 0.22:
                            final_beats.append(float(onsets_arr[nearest_idx]))
                        else:
                            final_beats.append(float(tap))
                else:
                    final_beats = raw_sorted

                final_beats = sorted(list(set(final_beats)))
                if final_beats[0] > 0.1:
                    final_beats.insert(0, 0.0)
                else:
                    final_beats[0] = 0.0

                if final_beats[-1] < total_duration:
                    final_beats.append(total_duration)

                self.set_status("Generiere Video aus manuellen Schnitten...", 0.1)
                render_settings = self.collect_render_settings()
                threading.Thread(
                    target=self.generate_edit,
                    args=(start_t, end_t, out_path, final_beats, "manual", master_audio_wav, False, None, render_settings),
                    daemon=True
                ).start()
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.set_status("Aufnahme abgebrochen (keine Schnitte erfasst).", 0.0)
                self.btn_generate.configure(state="normal")
                self.btn_cancel.configure(state="disabled")

        rec_win.protocol("WM_DELETE_WINDOW", on_close)

    # =====================================================================
    # Main Generation Pipeline (Pro Studio Engine)
    # =====================================================================
    def generate_edit(self, start_t, end_t, out_path, custom_beat_times=None, mode="auto", pre_master_audio=None, proxy=False, batch_label=None, settings=None):
        temp_dir = tempfile.mkdtemp(prefix="autoedit_render_")
        temp_files = []
        try:
            cfg = settings if settings else self.collect_render_settings()
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
            video_paths = cfg["video_paths"]

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
            fx_style = cfg["fx_style"]
            intensity = cfg["fx_intensity"]
            allow_shake = cfg["allow_shake"]
            allow_bloom = cfg["allow_bloom"]
            allow_punch = cfg["allow_punch"]
            allow_push = cfg["allow_push"]
            allow_rgb = cfg["allow_rgb"]
            allow_vignette = cfg["allow_vignette"]
            allow_speedramp = cfg["allow_speedramp"]
            allow_bass_pulse = cfg["allow_bass_pulse"]

            # Impact-Cadence: wie oft die Flash-/Shake-Impacts feuern
            cadence_step = 4
            if "Jeder Cut" in cfg["fx_cadence"]:
                cadence_step = 1
            elif "Jeder 2. Cut" in cfg["fx_cadence"]:
                cadence_step = 2

            # Jitter-Korrektur & Drop-Pre-Roll
            enable_snap = cfg["enable_snap"]
            preroll_choice = cfg["preroll_choice"]
            ai_audio_analysis = cfg["ai_audio_analysis"]
            overlay_manual_sec = cfg["overlay_manual_sec"]

            # KI-Auto-Reframe
            enable_auto_reframe = cfg["ai_reframe"] and orientation == "portrait"

            # Proxy-Modus (Schnell-Vorschau): kleines Format, wenig FPS, keine teuren Effekte
            if proxy:
                fps = 24
                crf_val = "30"
                preset_val = "ultrafast"
                res_choice = "Proxy (480p)"
                allow_shake = False
                allow_rgb = False
                allow_speedramp = False
                enable_auto_reframe = False
                apply_cinematic_lut = False
                allow_bass_pulse = False

            # Spinning Overlay Variables
            overlay_img = cfg["overlay_img_path"]
            has_overlay = cfg["enable_overlay"] and overlay_img and os.path.exists(overlay_img)
            overlay_start_type = cfg["overlay_start_type"]
            overlay_corner_mode = cfg["overlay_corner_mode"]
            overlay_size_str = cfg["overlay_size_str"]
            overlay_speed_val = cfg["overlay_speed_val"]
            overlay_opacity_val = cfg["overlay_opacity_val"]
            overlay_circular = cfg["overlay_circular"]

            # Master audio extraction
            master_audio_path = os.path.join(temp_dir, "master_audio.wav")
            if pre_master_audio and os.path.exists(pre_master_audio):
                shutil.copy(pre_master_audio, master_audio_path)
            else:
                cmd_audio = [
                    self.ffmpeg_exe, "-y",
                    "-ss", f"{start_t:.3f}",
                    "-i", audio_source_path,
                    "-t", f"{total_duration:.3f}",
                    "-vn",
                    "-c:a", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    master_audio_path
                ]
                subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if self.is_cancelled:
                raise InterruptedError("Vorgang abgebrochen.")

            # -------------------------------------------------------------
            # STEP 1: Multi-Band Transient Audio-Analyse & Drop-Erkennung
            # -------------------------------------------------------------
            drop_timestamp = 0.0
            if custom_beat_times is None:
                self.set_status("Schritt 1/3: Multi-Band Audio-Analyse & Transient-Tracking...", 0.08)
                sr_target = 22050
                y, sr = librosa.load(master_audio_path, sr=sr_target)
                y_percussive = librosa.effects.percussive(y, margin=2.2)

                env_low = librosa.onset.onset_strength(y=y_percussive, sr=sr, fmax=280)
                env_high = librosa.onset.onset_strength(y=y_percussive, sr=sr, fmin=800, fmax=5500)
                onset_env = env_low * 1.0 + env_high * 0.65

                kick_env = None
                kick_sr = sr
                kick_hop = 512

                if mode == "onsets":
                    self.set_status("Schritt 1/3: Erfasse dynamische Onset-Peaks...", 0.12)
                    hop_len = 512
                    onset_frames = librosa.onset.onset_detect(
                        onset_envelope=onset_env, sr=sr, hop_length=hop_len,
                        backtrack=True, wait=6, delta=0.07
                    )
                    raw_beats = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_len)
                    beat_times = list(raw_beats)

                    if enable_snap:
                        beat_times = snap_beats_to_transients(
                            beat_times, onset_env, sr, hop=hop_len, max_shift=0.06
                        )

                    if allow_bass_pulse:
                        kick_env = np.asarray(env_low, dtype=np.float32)
                        kick_env = kick_env / (float(np.percentile(kick_env, 98)) + 1e-9)

                else:
                    self.set_status("Schritt 1/3: Berechne Multi-Band Takt & Backtrack-Attack...", 0.12)
                    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                    beat_frames = librosa.onset.onset_backtrack(beat_frames, energy=onset_env)
                    raw_beats = librosa.frames_to_time(beat_frames, sr=sr)

                    rms = librosa.feature.rms(y=y)[0]
                    mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.01

                    # KI-basierte Beat-Drop & Struktur-Erkennung
                    try:
                        if ai_audio_analysis:
                            drop_raw_time = ai_detect_beat_drop(y, sr, total_duration)
                        else:
                            drop_raw_time = 0.0

                        if drop_raw_time > 0.05 and len(raw_beats) > 0:
                            nearest_idx = int(np.argmin(np.abs(np.array(raw_beats) - drop_raw_time)))
                            drop_timestamp = float(raw_beats[nearest_idx])
                        else:
                            drop_timestamp = max(0.0, drop_raw_time)
                    except Exception:
                        drop_timestamp = 0.0

                    beat_times = [float(b) for b in raw_beats]
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

                    # Jitter-Korrektur: Beats auf den naechsten Transient snappen
                    if enable_snap:
                        beat_times = snap_beats_to_transients(
                            beat_times, onset_env, sr, hop=512, max_shift=0.06
                        )

                    # Kick-Envelope fuer den Bass-Pulse Effekt (globaler Zeitindex)
                    if allow_bass_pulse:
                        try:
                            kick_env = np.asarray(env_low, dtype=np.float32)
                            kick_env = kick_env / (float(np.percentile(kick_env, 98)) + 1e-9)
                            kick_hop = 512
                        except Exception:
                            kick_env = None
                    else:
                        kick_env = None

                del y, y_percussive, env_low, env_high, onset_env

                beat_times = sorted([b for b in beat_times if 0.0 < b < total_duration])
                beat_times.insert(0, 0.0)
                beat_times.append(total_duration)

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
                if sanitized_beats[-1] != total_duration:
                    sanitized_beats[-1] = total_duration
                beat_times = sanitized_beats
            else:
                self.set_status("Schritt 1/3: Verwende manuelle Schnitte...", 0.15)
                beat_times = list(custom_beat_times)
                drop_timestamp = 0.0
                kick_env = None
                kick_hop = 512

            if self.is_cancelled:
                raise InterruptedError("Vorgang abgebrochen.")

            # -------------------------------------------------------------
            # Drop-Pre-Roll: Edit startet N Beats vor dem Beat-Drop.
            # Der Audio-Start wird passend mitverschoben (edit_offset).
            # -------------------------------------------------------------
            edit_offset = 0.0
            preroll_beats = 0
            if preroll_choice.startswith("2 Beats"):
                preroll_beats = 2
            elif preroll_choice.startswith("4 Beats"):
                preroll_beats = 4
            elif preroll_choice.startswith("8 Beats"):
                preroll_beats = 8
            elif preroll_choice.startswith("16 Beats"):
                preroll_beats = 16
            elif preroll_choice.startswith("32 Beats"):
                preroll_beats = 32

            if preroll_beats > 0 and drop_timestamp > 0.2 and mode != "manual":
                inner = [beat_times[k + 1] - beat_times[k] for k in range(len(beat_times) - 1)]
                inner = [d for d in inner if 0.15 < d < 2.0]
                beat_len = float(np.median(inner)) if inner else 0.5
                target = drop_timestamp - (preroll_beats * beat_len)
                if target <= 0.3:
                    # Pre-Roll reicht ueber den Trim-Start hinaus: das komplette
                    # Intro bleibt drin, es wird nur nichts abgeschnitten.
                    print(f"[Pre-Roll] {preroll_beats} Beats ueber Trim-Start hinaus - ganzes Intro bleibt")
                    edit_offset = 0.0
                elif target < total_duration - 1.0:
                    edit_offset = target
                    beat_times = [0.0] + [b - edit_offset for b in beat_times if b > edit_offset + 0.02]
                    drop_timestamp -= edit_offset
                    print(f"[Pre-Roll] {preroll_beats} Beats ({preroll_beats * beat_len:.2f}s) vor Drop, Offset {edit_offset:.2f}s")

            effective_duration = total_duration - edit_offset

            # Frame Quantization (Zero Accumulated Drift)
            raw_frame_indices = [int(round(t * fps)) for t in beat_times]
            clean_frame_indices = [raw_frame_indices[0]]
            min_cut_frames = 5 if fps >= 60 else 3
            for fi in raw_frame_indices[1:]:
                if fi - clean_frame_indices[-1] >= min_cut_frames:
                    clean_frame_indices.append(fi)
            total_target_frames = int(round(effective_duration * fps))
            clean_frame_indices[-1] = total_target_frames
            clean_frame_indices[0] = 0

            frame_counts = [clean_frame_indices[k + 1] - clean_frame_indices[k] for k in range(len(clean_frame_indices) - 1)]
            durations = [fc / float(fps) for fc in frame_counts]

            if has_overlay:
                if overlay_start_type == "start":
                    active_overlay_start = 0.0
                elif overlay_start_type == "drop":
                    active_overlay_start = drop_timestamp if drop_timestamp > 0 else max(0.0, overlay_manual_sec - edit_offset)
                else:
                    active_overlay_start = max(0.0, min(effective_duration, overlay_manual_sec - edit_offset))
            else:
                active_overlay_start = 999999.0

            if has_overlay:
                print(f"[Overlay] Aktiv ab {active_overlay_start:.2f}s (Trigger: {overlay_start_type}, Drop: {drop_timestamp:.2f}s)")

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
                cw, ch = clip.size
                if scale_mode == "fit":
                    scale = min(target_w / float(cw), target_h / float(ch))
                    resized = clip.resize(scale)
                    return resized.on_color(size=(target_w, target_h), color=(0, 0, 0), pos="center")

                scale = max(target_w / float(cw), target_h / float(ch)) * 1.002
                resized = clip.resize(scale)
                rw, rh = resized.size
                half_w = target_w / 2.0

                if reframer is None:
                    return resized.crop(
                        x_center=rw / 2.0,
                        y_center=rh / 2.0,
                        width=target_w,
                        height=target_h,
                    )

                # KI-Auto-Reframe: horizontal verschobener Crop, der dem Gesicht folgt
                def reframe_picker(gf, t):
                    frame = gf(t)
                    ratio = reframer.track_center(frame)
                    if ratio is None:
                        cx = rw / 2.0
                    else:
                        cx = ratio * rw
                        cx = max(half_w, min(rw - half_w, cx))
                    left = int(round(cx - half_w))
                    top = int(round((rh - target_h) / 2.0))
                    left = max(0, min(rw - target_w, left))
                    top = max(0, max(0, rh - target_h))
                    return frame[top:top + target_h, left:left + target_w]

                return resized.fl(reframe_picker)

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
                    cs = random.uniform(0, max_s) if max_s > 0 else 0.0
                end_s = min(cs + dur, safe_max_dur)
                sc = src.subclip(cs, end_s)
                if path is not None:
                    next_pos = end_s if end_s < safe_max_dur else 0.0
                    video_positions[path] = next_pos
                if sc.duration < dur:
                    sc = sc.fx(vfx.loop, duration=dur)
                return sc

            # =========================================================
            # Modern High-Impact Edit Effect Engine (Rhythm-Aware)
            # =========================================================
            def apply_pro_effects(clip, dur, cut_idx=0, cut_global_start=0.0):
                # Effekt-Dauern an die Cut-Laenge koppeln: bei schnellen
                # 2x-Edits darf der Flash nicht in den naechsten Beat laufen,
                # sonst sieht der Schnitt optisch aus dem Takt gerutscht.
                fd = min(0.09, dur * 0.22)
                sd = min(0.10, dur * 0.34)
                rd = min(0.06, dur * 0.18)
                pz = min(0.22, dur * 0.6)

                kick_strength = 0.0
                if allow_bass_pulse and kick_env is not None:
                    try:
                        k_idx = int(round((cut_global_start * kick_sr) / kick_hop))
                        if 0 <= k_idx < len(kick_env):
                            # Der Kick kommt erst NACH dem Cut-Punkt (onset_backtrack
                            # legt den Schnitt vor den Transient), also Vorwaertsfenster.
                            k_hi = min(len(kick_env), k_idx + 8)
                            kick_strength = float(np.clip(np.max(kick_env[k_idx:k_hi]), 0.0, 1.0))
                    except Exception:
                        kick_strength = 0.0

                # Impact-Cadence: Impact-Effekte nicht auf jedem Cut ausloesen
                cadence_hit = (cut_idx % cadence_step) == 0

                # Determine Motion Cadence for this cut:
                if "Dezent" in fx_style:
                    motion_type = 1 if (cut_idx % 2 == 0) else 3
                elif "Volle Power" in fx_style or "Hardcore" in fx_style:
                    motion_type = 0
                elif "Zufällig" in fx_style:
                    motion_type = random.choice([0, 1, 2, 3])
                else:
                    # Dynamischer Rhythmus-Flow (Abwechslungsreich):
                    # 0 = Downbeat Heavy (Tilt Shake + Punch + Flash)
                    # 1 = Crisp Optical Punch-Zoom (NO wobble!) + Flash
                    # 2 = Directional Pan Snap (Snap slide, NO tilt wobble) + RGB
                    # 3 = Cinematic Slow Push-In (Smooth glide, NO shake)
                    motion_type = cut_idx % 4

                # Alternating direction signs
                tilt_sign = 1.0 if ((cut_idx // 2) % 2 == 0) else -1.0
                pan_sign = 1.0 if (cut_idx % 2 == 0) else -1.0

                use_punch = allow_punch and (motion_type in [0, 1])
                use_shake = allow_shake and cadence_hit and (motion_type == 0) and (dur >= 0.22)
                use_snap = cadence_hit and (motion_type == 2) and (dur >= 0.18)
                use_bloom = allow_bloom and cadence_hit and (motion_type in [0, 1, 2])
                use_rgb = allow_rgb and cadence_hit and (motion_type in [0, 2])
                use_push = allow_push and (motion_type == 3 or dur > 0.60)

                def process_frame(gf, t):
                    frame = gf(t)

                    # 1. Spatial Transformation
                    decay_zoom = ((max(0.0, 1.0 - (t / max(0.001, pz)))) ** 3.0)
                    cur_zoom = 1.0
                    if use_punch:
                        cur_zoom += (0.11 * intensity * decay_zoom)
                    if use_push:
                        cur_zoom += (0.038 * (t / dur))

                    ang = 0.0
                    dx, dy = 0.0, 0.0

                    # Heavy Recoil Shake with alternating Tilt (Downbeats only!)
                    if use_shake and t < sd:
                        decay_shake = ((max(0.0, 1.0 - (t / sd))) ** 2.4)
                        ang = np.sin(t * 36.0) * (2.2 * intensity * tilt_sign) * decay_shake
                        dx = np.sin(t * 48.0) * (14.0 * intensity * pan_sign) * decay_shake
                        dy = np.cos(t * 38.0) * (10.0 * intensity) * decay_shake

                    # Directional Snap (Fast horizontal whip, zero tilt!)
                    elif use_snap and t < 0.12:
                        decay_snap = ((max(0.0, 1.0 - (t / 0.12))) ** 3.0)
                        dx = (20.0 * intensity * pan_sign) * decay_snap

                    if abs(ang) > 0.01 or cur_zoom > 1.002 or abs(dx) > 0.5 or abs(dy) > 0.5:
                        M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), ang, cur_zoom)
                        M[0, 2] += dx
                        M[1, 2] += dy
                        frame = cv2.warpAffine(frame, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                    # 2. Exposure Bloom Flash (scharfe Spitze auf dem Transient)
                    if use_bloom and t < fd:
                        decay_fl = ((max(0.0, 1.0 - (t / fd))) ** 3.0) * intensity
                        gain = 1.0 + (0.85 * decay_fl)
                        warm_tint = np.array([255, 246, 225], dtype=np.float32)
                        boosted = frame.astype(np.float32) * gain + (warm_tint * (0.25 * decay_fl))
                        frame = np.clip(boosted, 0, 255).astype(np.uint8)

                    # 3. Radial Lens Chromatic Aberration (RGB-Linsenzerlegung)
                    if use_rgb and t < rd:
                        decay_rgb = ((max(0.0, 1.0 - (t / rd))) ** 2.0) * intensity
                        if decay_rgb > 0.02:
                            r_scale = 1.0 + (0.014 * decay_rgb)
                            b_scale = 1.0 - (0.014 * decay_rgb)
                            Mr = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), 0, r_scale)
                            Mb = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), 0, b_scale)
                            r_ch = cv2.warpAffine(frame[:, :, 0], Mr, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                            b_ch = cv2.warpAffine(frame[:, :, 2], Mb, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                            frame[:, :, 0] = r_ch
                            frame[:, :, 2] = b_ch

                    # 4. Bass-Pulse (Full-Frame Kick-Darkening)
                    if kick_strength > 0.12 and t < 0.16:
                        bp = (max(0.0, 1.0 - (t / 0.16)) ** 2.0) * kick_strength * intensity
                        if bp > 0.015:
                            dim = 1.0 - (0.34 * bp)
                            tint = np.array([1.0 - (0.18 * bp), 1.0 - (0.30 * bp), 1.0 - (0.05 * bp)], dtype=np.float32)
                            frame = np.clip(frame.astype(np.float32) * dim * tint, 0, 255).astype(np.uint8)

                    # 5. Cinematic Vignette
                    if allow_vignette and vignette_mask is not None:
                        frame = cv2.multiply(frame, vignette_mask, scale=1.0 / 255.0)

                    # 6. Cinematic S-Curve Contrast & Color Pop LUT
                    if apply_cinematic_lut:
                        frame = cv2.LUT(frame, CINEMATIC_LUT)

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

            cpu_threads = max(4, min(16, os.cpu_count() or 4))
            clip_idx = 0
            i = 0
            total_dur_count = len(durations)
            clip_loop_t0 = time.time()
            prefix = f"{batch_label} | " if batch_label else ""

            while i < total_dur_count:
                if self.is_cancelled:
                    raise InterruptedError("Vorgang durch Benutzer abgebrochen.")

                duration = float(durations[i])
                if duration < 0.10:
                    duration = 0.10

                cut_global_start = clean_frame_indices[i] / float(fps)
                cur_path = video_paths[clip_idx]
                source_clip = open_source(cur_path)

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

                # Apply Pro Effects with diverse rhythmic cadence
                sub_clip = apply_pro_effects(sub_clip, duration, cut_idx=i, cut_global_start=cut_global_start)

                # Apply Spinning Overlay if clip overlaps or is past active_overlay_start
                if has_overlay and (cut_global_start + duration >= active_overlay_start):
                    sub_clip = apply_spinning_overlay(sub_clip, i, cut_global_start, duration)

                # Audio handling
                seg_has_audio = False
                if clip_audio_vol > 0.0:
                    try:
                        if raw_cut.audio is not None:
                            sub_clip = sub_clip.set_audio(raw_cut.audio.volumex(clip_audio_vol))
                            seg_has_audio = True
                    except Exception:
                        seg_has_audio = False

                if not seg_has_audio:
                    sub_clip = sub_clip.without_audio()

                temp_path = os.path.join(temp_dir, f"seg_{len(temp_files):04d}.mp4")
                write_args = {
                    "fps": fps,
                    "codec": "libx264",
                    "threads": cpu_threads,
                    "logger": None,
                    "ffmpeg_params": [
                        "-preset", preset_val,
                        "-crf", crf_val,
                        "-tune", "film",
                        "-pix_fmt", "yuv420p"
                    ]
                }
                if seg_has_audio:
                    write_args["audio_codec"] = "aac"

                sub_clip.write_videofile(temp_path, **write_args)
                temp_files.append(temp_path)

                source_clip.close()
                try:
                    sub_clip.close()
                except Exception:
                    pass

                clip_idx = (clip_idx + 1) % len(video_paths)
                i += 1

                pct = 0.20 + (0.68 * (i / total_dur_count))
                self.set_status(f"{prefix}Schritt 2/3: Erstelle Beat-Clip {len(temp_files)}/{total_dur_count} ({fps} FPS, {tw}x{th})...", pct)
                if i < total_dur_count:
                    elapsed = time.time() - clip_loop_t0
                    eta = (elapsed / i) * (total_dur_count - i)
                    self.set_eta(f"Restzeit Clip-Render: {format_eta(eta)}")
                else:
                    self.set_eta("Restzeit Clip-Render: 0s")

            if self.is_cancelled:
                raise InterruptedError("Vorgang durch Benutzer abgebrochen.")

            # -------------------------------------------------------------
            # STEP 3: Ultra-Fast Concat + Studio Master Audio Mux (320 kbps)
            # -------------------------------------------------------------
            self.set_status("Schritt 3/3: Finale Studio-Zusammensetzung & Audio-Muxing...", 0.90)

            concat_txt = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_txt, "w", encoding="utf-8") as f:
                for tp in temp_files:
                    f.write(f"file '{os.path.abspath(tp).replace(os.sep, '/')}'\n")

            visual_concat_mp4 = os.path.join(temp_dir, "visual_concat.mp4")
            cmd_concat = [
                self.ffmpeg_exe, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_txt,
                "-c", "copy",
                visual_concat_mp4
            ]
            res_c = subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res_c.returncode != 0:
                cmd_concat_fallback = [
                    self.ffmpeg_exe, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_txt,
                    "-c:v", "libx264",
                    "-preset", preset_val,
                    "-crf", crf_val,
                    visual_concat_mp4
                ]
                subprocess.run(cmd_concat_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            self.set_status(f"{prefix}Schritt 3/3: Muxe unkomprimiertes Master-Audio & Fade-Out...", 0.95)

            audio_filters = []
            if edit_offset > 0.01:
                audio_filters.append(f"atrim=start={edit_offset:.3f},asetpts=PTS-STARTPTS")
            if enable_fade_out:
                audio_filters.append(f"afade=t=out:st={max(0.0, effective_duration - 0.8):.2f}:d=0.8")
            if enable_auto_loudness:
                audio_filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
            audio_filter = ",".join(audio_filters) if audio_filters else "anull"

            final_cmd = [
                self.ffmpeg_exe, "-y",
                "-i", visual_concat_mp4,
                "-i", master_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "320k",
                "-t", f"{effective_duration:.3f}",
                "-af", audio_filter,
                "-shortest",
                out_path
            ]
            subprocess.run(final_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            self.last_rendered_path = out_path
            self.set_eta("")
            self.set_status(f"Export erfolgreich abgeschlossen: {os.path.basename(out_path)}", 1.0)
            if not batch_label:
                self.show_render_done(out_path)
            self._ui_after(0, lambda: self.post_action_frame.pack(fill="x", padx=20, pady=(6, 10)))

        except InterruptedError:
            self.set_status("Rendern abgebrochen. Keine Datei erstellt.", 0.0)
        except Exception as e:
            traceback.print_exc()
            self.set_status(f"Fehler: {e}", 0.0)
            self.show_error("Fehler", f"Fehler beim Rendern:\n{traceback.format_exc()}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.set_eta("")
            self.render_finished.set()
            self._ui_after(0, lambda: self._set_render_buttons(False))


if __name__ == "__main__":
    app = EditMakerApp()
    app.mainloop()
