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
- **Effekt-Engine (CapCut-Stil, bewusst sparsam)**:
  - Die Mehrheit der Cuts bleibt voellig sauber. Effekte landen auf musikalischen
    Positionen - Drop, Akzente, Cadence - und nicht auf jedem Schnitt. Genau das
    war der Unterschied zwischen "sieht aus wie ein Beat-Edit" und "sieht aus wie
    eine PowerPoint-Folienfolge". Gemessen: 62-90% der Cuts bleiben unberuehrt.
  - **Whip Pan** mit echter Bewegungsunschaerfe (Linienkernel) als Uebergang, Richtungen alternierend.
  - **Punch In/Out** mit Smoothstep-Ease, Peak exakt auf dem Transient, Abklingen in <=200ms.
  - **Zoom-Blur-Burst** und optionaler **Freeze-Frame** genau auf dem Beat-Drop.
  - **Filmkorn** als statisches Feld. Pro Frame gewuerfelt wuerde die Interframe-Kompression
    zerstoeren und dieselbe Renderzeit um das 8-Fache aufblaehen.
  - Kein Dauer-Zoom mehr: die alte Engine transformierte jeden Frame, dadurch wirkte
    jedes Video wie eine langsame Diashow.
  - Fuenf Effekt-Sets: CapCut Kinetic, Cinematic, Beat Edit, Maximal, Clean Cut.
- **Collage & Spin** (aus dem Original-Repo):
  - 4er-Raster-Collage (Hintergrund + 4 Kacheln, Beat fuer Beat einblendend) und 16-Raster (4x4).
  - Spin mit 360-Grad-Drehung ueber die Clipdauer.
  - Beide verbrauchen exakt die Beats, die sie belegen - das Beat-Gitter bleibt synchron.
- **Reproduzierbare Renders**:
  - Effekt-Placement und Quell-Ausschnitte werden aus einem CRC32-Seed aus Song + Trim abgeleitet.
  - Gleiche Musik + gleicher Trim => identisches Ergebnis, unabhaengig vom Ausgabedateinamen.
- **Praeziser Beat-Sync & Takt-Magnet**:
  - Multi-Band Transient Tracking (Kicks und Snares getrennt) mit Attack-Backtracking.
  - Manuelle Klicks rasten ueber den Takt-Magneten praezise an den echten Drum-Peaks ein.
  - Frame-Quantisierung verhindert jeglichen kumulierten Drift ueber die gesamte Songlaenge.
- **Impact-Rhythmus**: Akzent-Effekte auf jedem, jedem 2. oder nur jedem 4. Cut.
- **Timing-kopplte Effekt-Dauern**: Bloom-Flash, Bass-Shake, RGB und Punch-Zoom skalieren mit der Laenge des Cuts. Ein 0.14s-Flash auf einem 0.28s-Cut (2x-Tempo) frisst den halben Beat und sieht dadurch aus wie ein Timing-Fehler.
- **Latenz-Kalibrierung fuer den manuellen Modus**:
  - Tastatur, Soundkarte, Bluetooth-Headset und Monitor bringen jeweils eigene Verzoegerung mit. Ohne Korrektur ist jeder manuelle Cut um genau diesen Betrag daneben.
  - "Kalibrieren" spielt 12 Klicks, der Nutzer tippt mit, und der Median der Differenz wird als Offset uebernommen und in `app_settings.json` gespeichert.
  - Latenz zusaetzlich per Slider feinjustierbar, Live-Anzeige zeigt bei jedem Tap den Abstand zum naechsten Beat (gruen/gelb/rot) und wie weit der naechste Beat weg ist.
  - Gemessen: Restfehler sinkt von 108 ms auf 0 ms.
- **Aufnahme-Review nach dem manuellen Schneiden**:
  - Nach dem Aufnehmen geht es nicht mehr direkt ins Rendern, sondern in eine Pruefansicht.
  - Wellenform des Auswahlbereichs mit einem Marker pro Cut (gruen = am Beat eingerastet, orange duenn = roher Tap) und Positionsanzeige waehrend der Wiedergabe.
  - **Audio-Indikatoren**: ein Klickton markiert jeden Cut hoerbar, damit man ohne Rendern hoert, ob der Schnitt sitzt. Abschaltbar.
  - Zusammenfassung "N Schnitte | X eingerastet | Y frei | Z%" plus Navigation (Von vorn / Zum ersten Cut).
  - "Zufrieden - rendern" startet den Export, "Neu aufnehmen" verwirft und macht den Recorder erneut auf.
- **KI Beat-Drop-Erkennung (Machine Learning)**:
  - Agglomerative Segmentierung (Chroma-CQT + MFCC) plus Multi-Band-Change-Point-Analyse.
  - Waehlt den Drop, der in den lautesten Abschnitt fuehrt - nicht den mit dem groessten relativen Sprung.
  - Intro-Regel: beginnt der Song still, gewinnt der erste echte Drop. Snapping auf den Transient.
  - Analysezeit 0.4-1.5s (ohne HPSS, das allein 10s pro Track gekostet haette).
- **Drop-Kandidaten-Analyse (zweite KI-Stufe)**:
  - "Drop-Kandidaten suchen" durchsucht ausschliesslich den markierten Auswahlbereich nach saemtlichen Drops.
  - Ergebnis-Fenster listet jeden Treffer mit Zeit, Konfidenz, Energiesprung und Kennzeichnung, ob die Intro-Regel gegriffen hat.
  - Der beste Kandidat wird automatisch gesetzt und violett als "BEST" markiert, alle weiteren als gestrichelte Linien - in der Waveform und in der Liste.
  - Zeile anklicken uebernimmt einen anderen Kandidaten. Bis zu 8 Kandidaten pro Bereich.
- **Fensterlayout**: zweispaltiges Dashboard (Eingabe links, Effekte rechts), Profil- und Renderleiste fixiert, Render-Buttons immer sichtbar statt im Scrollbereich versteckt.
- **Drop-Pre-Roll / Intro**: Edit startet auf Wunsch 2, 4, 8, 16 oder 32 Beats vor dem Drop. Reicht der Pre-Roll ueber den Trim-Start hinaus, bleibt das komplette Intro erhalten. Audio wird per `atrim/asetpts` passend mitverschoben, Bild und Ton bleiben synchron.
- **Impact-Rhythmus**: Akzent-Effekte auf jedem, jedem 2. oder nur jedem 4. Cut.
- **Timing-kopplte Effekt-Dauern**: Bloom-Flash, Bass-Shake, RGB und Punch-Zoom skalieren mit der Laenge des Cuts. Ein 0.14s-Flash auf einem 0.28s-Cut (2x-Tempo) frisst den halben Beat und sieht dadurch aus wie ein Timing-Fehler.
- **Jitter-Korrektur**: Snappt jeden Beat auf den naechsten echten Transient (+-60ms) - kein Flackern mehr im 2x-Tempo.
- **Best-Moment-Finder**: findet die energiereichste Passage (15/30/60/90s) und legt den Hook so, dass der Drop in den ersten Sekunden landet.
- **Wellenform-Ansicht**: Vollstaendige Waveform mit Beat-Gitter, Trim-Bereich und rotem DROP-Marker.
  - **Linksklick** setzt den Startpunkt, **Rechtsklick** setzt den Endpunkt.
  - Beim Vorhoeren laeuft ein gelber Playhead durch die Waveform, die aktuelle Position steht rechts daneben.
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
  - Fertig-Dialog mit **"Direkt ansehen"**, **"Ordner oeffnen"** und **"Schliessen"** statt blockierender Meldungsbox.
  - Restzeit-ETA im Statusbereich waehrend des Clip-Renders.

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
