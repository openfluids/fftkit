# README Banner v2

Asset: `assets/readme-banner-v2.jpg` (1408x469, 3:1, 210 KB)

Tool/model: xAI Grok CLI, built-in `image_gen` tool, plus local compositing.

Replaces `readme-banner-v1.png` (commit 5591760), which was a flat two-panel
diagram — a waveform on the left, a spectrum on the right, wordmark between
them. It was correct and dull: thin flat strokes, a dead centre, and nothing
that made anyone want to look twice.

## Approach

The wordmark is **not** generated. Image models render short lowercase words
unpredictably, and accepting whatever letterforms come back is most of what
makes a generated banner look cheap. So the artwork is generated deliberately
textless, and the type is set locally afterwards in Lato Light with controlled
tracking. That also means the wordmark can be re-set — different weight, size,
position — without regenerating the art.

## Prompt (artwork only, no text)

```text
A stunning abstract scientific artwork, wide 2:1 landscape, for use as a premium
software banner. A three-dimensional spectrogram waterfall: dozens of glowing
spectral ridgelines stacked one behind another, receding into deep perspective
toward a vanishing point on the right, each ridgeline a jagged luminous profile
of peaks and valleys. The nearest ridges are sharp and brilliant electric cyan;
the ones further back fade through teal into deep indigo darkness with
atmospheric haze and depth of field. A few dominant peaks flare hot coral and
amber, casting soft volumetric glow into the surrounding dark. Background is a
rich near-black charcoal with a subtle vertical gradient and fine film grain.
Cinematic lighting, bloom, high dynamic range, deep blacks, luminous highlights.
Extremely refined, expensive-looking, gallery quality, sophisticated data-art
aesthetic. ABSOLUTELY NO TEXT, no letters, no words, no numbers, no labels, no
axes, no logos, no watermarks, no UI elements. Pure abstract imagery only. Leave
the left third of the frame comparatively dark, calm and uncluttered as negative
space.
```

The subject is not decoration: a spectrogram waterfall is what this library
produces. Each ridgeline is one spectrum, the stack is time, and the peaks
flaring coral and amber are resolved modes.

## Post-processing

- `image_gen` rejects a 3:1 request and falls back to 2:1, so the returned
  1408x704 image was cropped to 1408x469 — matching the 3:1 aspect of the
  sibling `openfluids/dynachaos` banner. The crop is biased upward (30% from the
  top) to keep the brightest ridgelines and leave calm space for the wordmark.
- Wordmark composited locally: Lato Light at 112 px, +7 px tracking, warm
  off-white `#F7F3EC`, placed at (80, 55) in the cropped frame. A wide, heavily
  blurred dark halo sits under the type so it stays legible without reading as a
  drop shadow.
- Saved as JPEG q95 with no chroma subsampling: 210 KB, against 720 KB for the
  equivalent PNG and 2.1 MB for the dynachaos banner. The image is a smooth
  gradient render with no flat colour fields, which is the case PNG handles
  worst and JPEG handles best. Inspected at full resolution for ringing around
  the letterforms; none is visible.

## Rejected alternatives

Ten images across four rounds. The three that reached the final comparison:

- **Wave interference** — translucent sine waves building into luminous
  caustics. The most on-brand palette of the three and the cleanest negative
  space, but less specific to what the library does.
- **Turbulent cascade** — large vortices breaking down into fine filaments.
  Beautiful, and the most fluid-dynamical, but reads as smoke art rather than
  signal analysis.
- **Radial burst** — rejected outright: a generic sci-fi hyperspace look with
  no connection to spectral analysis.

An earlier round produced a clean image whose "spectrum" was a **sigmoid**
rather than a power law. On a library about spectral slopes that is a visible
error, so every later prompt states that the decay must be a straight line and
never an S-curve.
