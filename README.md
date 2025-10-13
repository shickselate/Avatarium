# 🎭 VR Companion — Interactive Scene Composer with ComfyUI & LLMs

**VR Companion** is an interactive, AI-driven character system combining real-time conversation, affective reasoning, and image generation via **ComfyUI**. The companion reacts to your chat by composing structured edit prompts that drive image-to-image updates—great for VR or browser-based storytelling.

---

## 🧠 Core Loop

```
User input → LLM interpretation → Structured edit plan → ComfyUI render → Live image refresh
```

- Short, in-character dialogue.
- Behind the scenes: explicit, sectioned edit instructions that guide the **Nunchaku Qwen Image Edit** workflow.

---

## ⚙️ Architecture

| Layer | Purpose |
|------|---------|
| **ComfyUI (Nunchaku Qwen Edit workflow)** | Diffusion-based image editing from structured prompts. |
| **Python orchestration** (`companion_web_v10x.py`) | Gradio UI + LLM planner + ComfyUI runner. |
| **Gradio web UI** | Chat, live image panel, “composed prompt” preview, heartbeat-driven refresh. |
| **Scene Composer** | Interprets intent → fills slots: Background, Face, Body, Hands/Props, **Wardrobe**, Camera/Light, Style. |
| **LLM backends** | OpenAI GPT‑4o‑mini or local Ollama models for dialogue + planning. |

---

## 🌟 Key Features

- **Structured Edit Prompting** (slot-based):  
  Clear sections (BACKGROUND / FACE / BODY / HANDS / **WARDROBE** / CAMERA+LIGHT / STYLE) that diffusion edit models follow reliably.
- **Scene–Character Fusion**:  
  Merges environment, gesture, mood, and optional backstory into a single, controllable prompt.
- **Continuity & Real-Time UX**:  
  Heartbeat polling keeps the last frame visible while rendering the next; choose Base / Last render / Custom as the next input.
- **Creative Controls**:  
  Per-aspect toggles and intensity sliders (0–2), “variation” for stronger background changes, live “Composed Prompt” preview + history log.
- **Multi-Backend Dialogue**:  
  Pluggable OpenAI or **Ollama**; JSON-structured planning for reproducibility.

---

## 🧰 Requirements

- Python **3.11+**
- Local **ComfyUI**
- Python libs: `gradio`, `requests`, `pillow`
- Optional: **OpenAI** API key or **Ollama** runtime for LLM responses

Quick install:
```bash
pip install gradio requests pillow
```

---

## 🚀 Run

1. Start **ComfyUI** locally (default REST: `http://127.0.0.1:8188`).
2. Ensure your workflow JSON for **Nunchaku Qwen Edit Lightning** is reachable (see UI field).
3. Launch the web app:
   ```bash
   python companion_web_v10e.py
   ```
4. Open the local Gradio URL (e.g., `http://127.0.0.1:7860`) and try:
   > “Let’s go to a gothic cathedral, candle in hand.”

You’ll see a short, in-character reply and a detailed **composed prompt** that updates the image.

---

## 🔧 Configuration (in the UI)

- **Input source**: Base image / Last render / Custom; optional upload of last render back to Comfy’s input.  
- **Scene Composer**: Creative boost, style preset (cinema/portrait/moody/fantasy), backstory, baseline mood, variation.  
- **Structured Edit Mode**:  
  - Per-slot toggles (Background, Body, Face, Hands, **Wardrobe**, Camera/Light, Style).  
  - Intensity sliders per aspect (0–2).  
  - Editable template with placeholders like `{{BACKGROUND}}`, `{{FACE}}`, etc.  
- **Planner & Gating**: Change-score Δ (0–4) vs last plan, threshold + cooldown, **Manual render** button.  
- **LLM**: OpenAI / Ollama / None; Ping button and visible status.

---

## 🧠 Technical Highlights

- Asynchronous Gradio chain: **Prime → Plan → Render** with chat state and no flicker.  
- **Heartbeat polling** of `./outputs` to keep last frame on-screen until the new image lands.  
- Hybrid **heuristics + LLM** for environment/gesture/mood/wardrobe inference.  
- Deterministic, transparent planning via **JSON** messages and a visible prompt preview.  
- Direct subprocess call to your `nunchaku_edit.py` driver (minimal coupling, no REST JSON templating required).

---

## 🪄 Example (Structured Edit Mode)

```
EDIT INSTRUCTIONS (clear, actionable):
BACKGROUND → gothic cathedral interior, stained glass windows, dust rays
FACE → soft smile, gaze right
BODY / GESTURE → relaxed posture, hands together
HANDS / PROPS → hold a small candle
WARDROBE → dark wool coat with subtle embroidery
CAMERA / LIGHT → close-up, cinematic lighting, volumetric rays
STYLE → cinematic lighting, filmic contrast
```

---

## 🧩 Roadmap

- Emotion + memory state machine for persistent personality.  
- VR front-end (A‑Frame/WebXR) and Meta Quest testing.  
- Direct ComfyUI queue integration + streaming progress.  
- Wardrobe presets & pose libraries.

---

## 🙌 Credits

Built by **Stephen Hicks** as part of ongoing research into **AI‑mediated emotional presence**, **neural storytelling**, and **interactive cinematic worlds**.

---

## 📄 License

MIT — see `LICENSE`.
