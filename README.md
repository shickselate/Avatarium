
# Companion Web v10f

**Companion Web** is an interactive AI-driven VR companion and visual storytelling interface.  
Version 10f integrates **OpenAI GPT‑4o‑mini** as the default conversational model and introduces stronger **head, face, and gesture control** for dynamic character animation through **ComfyUI** workflows.

---

## 🌟 Features

### Conversational AI
- Default backend: **OpenAI GPT‑4o‑mini** (requires `OPENAI_API_KEY`).
- Optional local backend: **Ollama** (e.g. `llama3.1:8b`).
- In‑character conversation with emotion, gestures, and scene awareness.
- Auto‑generation of structured visual prompts describing **pose, gaze, hands, wardrobe, and environment**.

### Enhanced Character Motion
- New expressivity sliders:
  - **Head movement** → look left/right/up/down, gentle tilt.
  - **Expression variance** → micro‑expressions and emotional shifts.
  - **Gesture energy** → talking‑hands emphasis and open‑palm dynamics.
- More vivid **HEAD / GAZE** and **HANDS / GESTURES / PROPS** sections in prompt composition.
- Automatically adjusts **camera framing** and **lighting** for cinematic continuity.

### Memory System
- YAML‑based long‑term memory (per‑character).
- Stores transcripts, gestures, episodes, and pinned facts.
- Automatically saved and reloaded on app restart.
- Manual “Save Memory”, “Pin Fact”, and “Summarize Session” buttons.

### Scene Composition & Rendering
- Generates structured edit instructions for **ComfyUI** via the `nunchaku_edit.py` workflow.
- Choice of image input source: Base, Last Render, or Custom.
- Automatic gating for render triggering (threshold + cooldown).
- Optional “Manual Render” override button.

### Album Integration
- Save current frame and metadata (“Save to Album”).
- Thumbnail gallery of previous renders with prompt captions.
- Memory entries record album saves as episodes.

### Live Preview & Heartbeat
- Image auto‑refreshes every second, maintaining continuity.
- Prevents blank screen between renders when ComfyUI output updates slowly.

---

## 🧠 Dependencies

| Component | Purpose |
|------------|----------|
| **Gradio** | Web UI and reactive state management |
| **Requests** | REST communication with OpenAI and ComfyUI |
| **PyYAML** | Persistent character memory storage |
| **ComfyUI** | Image generation workflow engine |
| **memory_manager.py** | Handles episodic / semantic memory |
| **album_manager.py** | Image gallery and metadata persistence |
| **nunchaku_edit.py** | Bridge script for ComfyUI edit workflows |

Optional: **Ollama** for local LLM execution.

Install with:
```bash
pip install gradio requests pyyaml pillow
```

---

## 🚀 Usage

1. Ensure your **ComfyUI** server and **nunchaku_edit.py** workflow are running.
2. Export your OpenAI key:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
3. Launch the companion:
   ```bash
   python companion_web_v10f.py
   ```
4. Open your browser at `http://127.0.0.1:7860`.
5. Chat with your character and watch her respond in dialogue and pose.

---

## 🧩 Project Structure

```
/companion_web_v10f.py      # Main app (Gradio frontend + LLM + ComfyUI orchestration)
/nunchaku_edit.py           # ComfyUI workflow bridge
/memory_manager.py          # Persistent character memories
/album_manager.py           # Local image album manager
/memory/                    # Character memory YAMLs
/outputs/                   # Rendered image outputs
/input/                     # Base reference images
```

---

## 🗂️ Version Highlights

**v10f (current)**
- Default OpenAI backend (gpt‑4o‑mini).
- Dramatically improved head, expression, and gesture dynamics.
- Added wardrobe and prop awareness.
- Optimized gating & heartbeat behaviour.
- Integrated album and memory persistence.

**v10e and earlier**
- Introduced memory persistence and structured edit templates.
- Added album saving, session summaries, and gating thresholds.

---

## 📄 License

This project is released under the **MIT License**.

---

## 💬 Acknowledgements

Developed collaboratively with iterative AI feedback by **Stephen Hicks** (Mendea / Fusion Shift projects).  
Inspired by cinematic embodiment, human–AI co‑presence, and VR storytelling research.
