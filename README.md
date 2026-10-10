# AutoEditMaker (Pro Studio Edition)

Ein professionelles Beat-Sync Video-Editing Tool mit moderner CustomTkinter-Oberflaeche, hardwarebeschleunigter FFmpeg-Engine, praeziser Librosa-Audioanalyse und High-Impact Videoeffekten.

## Highlights & Features

- **Profile & Presets (Speichern & Laden)**:
  - Vollstaendige Presets direkt speichern, laden, importieren und exportieren.
  - Vorkonfigurierte Profile fuer TikTok/Reels, Velocity, Cinematic und Minimal.
- **Spinning Bild-Overlay**:
  - Eigenes Bild (PNG mit Transparenz oder JPG) einbinden.
  - Kontinuierliche Rotation ueber Schnittgrenzen hinweg mit Geschwindigkeits- und Deckkraft-Regler.
  - Start-Trigger: Automatisch ab dem Beat-Drop, ab einer festgelegten Sekunde oder von Anfang an.
  - Positionierung: Animierter Eckenwechsel zum Takt, feste Ecken oder zentriert in der Bildmitte.
  - Optionaler Kreiszuschnitt mit weichem Rand.
- **60 FPS Support & fluessiges Playback**:
  - Waehlbar zwischen 60 FPS und 30 FPS fuer fluessige Action- und Gameplay-Edits.
- **Professionelle Impact-Effekte (Keine billigen Glitches)**:
  - **Bass-Shake**: Gedaempfter Kamera-Rueckstoss mit dynamischem Rotations-Tilt auf Kick-Hits.
  - **Exposure Bloom-Flash**: Echter Highlight-Lichtburst mit Belichtungskurve (kein Milcheffekt).
  - **Punch-Zoom**: Dynamischer optischer Beat-Punch mit kubischem Ease-Out.
  - **Radial RGB-Aberration**: Optische chromatische Linsenzerlegung an den Bildraendern.
  - **Cinematic Vignette**: Dunkler Randgradient fuer Fokussierung auf die Bildmitte.
  - **Cinematic S-Curve LUT**: Erhoehter Kontrast und Farbsaettigung fuer satten Look.
  - **Slow Push-In**: Sanfte filmische Kamerafahrt fuer ruhigere Passagen.
  - **Speed-Ramp**: Taktgenaue Geschwindigkeitsrampen.
  - **Effekt-Intensitaet**: Stufenlos von 20% bis 200% anpassbar.
- **Praeziser Beat-Sync & Takt-Magnet**:
  - Multi-Band Transient Tracking (Kicks und Snares getrennt) mit Attack-Backtracking.
  - Manuelle Klicks rasten ueber den Takt-Magneten praezise an den echten Drum-Peaks ein.
  - Frame-Quantisierung verhindert jeglichen kumulierten Drift ueber die gesamte Songlaenge.
- **KI Beat-Drop-Erkennung (Machine Learning)**:
  - Agglomerative Segmentierung (Chroma-CQT + MFCC) plus Multi-Band-Change-Point-Analyse.
  - Waehlt den Drop, der in den lautesten Abschnitt fuehrt - nicht den mit dem groessten relativen Sprung.
  - Intro-Regel: beginnt der Song still, gewinnt der erste echte Drop. Snapping auf den Transient.
  - Analysezeit 0.4-1.5s (ohne HPSS, das allein 10s pro Track gekostet haette).
- **Drop-Pre-Roll**: Edit startet auf Wunsch 2, 4 oder 8 Beats vor dem Drop (Audio wird passend mitverschoben).
- **Jitter-Korrektur**: Snappt jeden Beat auf den naechsten echten Transient (+-60ms) - kein Flackern mehr im 2x-Tempo.
- **Best-Moment-Finder**: findet die energiereichste Passage (15/30/60/90s) und legt den Hook so, dass der Drop in den ersten Sekunden landet.
- **Wellenform-Ansicht**: Vollstaendige Waveform mit Beat-Gitter, Trim-Bereich und rotem DROP-Marker. Klick setzt den Startpunkt.
- **KI-Auto-Reframe**: haelt das Subject im 9:16-Crop (YuNet-Gesichtserkennung, sonst Hautton-/Bewegungs-Tracking, zeitlich geglaettet).
- **Schnell-Vorschau (Proxy)**: 480p/24fps-Proxy-Render in ~2.5x kuerzerer Zeit, um Settings vor dem Final-Render zu testen.
- **Batch-Export**: rendert N Varianten mit gemischter Clip-Reihenfolge in einem Durchlauf.
- **Projekt-Speichern**: kompletter Arbeitsstand (Clips, Song, Trim, Effekte, Overlay) als `.aep`, mit relativen Pfaden.
- **Undo/Redo**: bis zu 60 Schritte ueber alle Einstellungen, Medien und Trim-Punkte.
- **Restzeit-Anzeige**: ETA im Statusbereich waehrend des Clip-Renders.
- **Bass-Pulse**: Full-Frame-Kick-Darkening, Staerke aus dem echten Low-Band-Onset-Env.
- **Auto-Loudness**: EBU R128 Normalisierung (-14 LUFS, True-Peak-Limiter) auf dem Master-Audio.
- **Audio-Mixing & Fade-Out**:
  - Original Clip-Sound stummschalten oder zu 15%/35% dezent beibehalten (fuer In-Game Sounds/SFX).
  - Automatischer 0.8s Fade-Out am Ende gegen abruptes Abreissen.
- **Prozesskontrolle & Schnellzugriff**:
  - Laufende Rendervorgaenge sauber per Knopfdruck abbrechen.
  - Nach Fertigstellung direkt den Ausgabeordner oeffnen oder das Video abspielen.

## Installation & Start

### Schnellstart (Windows)
Doppelklick auf `start.bat`. Das Skript erstellt bei Bedarf eine virtuelle Umgebung (`venv`), installiert alle noetigen Abhaengigkeiten und startet die App.

### Manuelle Installation

1. Virtuelle Umgebung anlegen & aktivieren:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

2. Abhaengigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

3. Anwendung starten:
   ```bash
   python app.py
   ```

## Optional: Gesichtserkennung fuer Auto-Reframe

OpenCV 5 hat die klassischen Haar-Cascades entfernt. Das Auto-Reframe nutzt
automatisch Hautton- und Bewegungs-Tracking. Wer eine praezisere
Gesichtserkennung will, legt einfach ein YuNet-Modell neben `app.py`:

```
app.py
face_detection_yunet_2023mar.onnx
```

Die App erkennt die Datei beim Start und schaltet dann auf die DNN-Erkennung um.
