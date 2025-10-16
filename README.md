# The Avatarium

**Avatarium** is an evolving AI–VR storytelling platform where characters from a living narrative world become responsive, expressive presences.  
Each avatar has a distinct backstory, memory, and agency, capable of perceiving, remembering, and reacting to the world — and to you.

Built as a modular, VR‑ready environment, Avatarium combines large‑language models, generative imagery, and character memory systems to create continuous, emotionally rich experiences. Every interaction can subtly reshape a character’s history and the shared fictional world.

---

<p align="center">
  <img src="./assets/soren01.png" alt="Soren smiling" width="45%"/>
  <img src="./assets/soren02.png" alt="Soren worried" width="45%"/><br/>
  <img src="./assets/soren03.png" alt="Soren standing at the beach" width="45%"/>
  <img src="./assets/soren04.png" alt="Soren at the beach, looking at a holographic display" width="45%"/>
</p>

<p align="center"><em>
Soren — a transhuman explorer shaped by oceanic evolution.  
His story unfolds through memory, perception, and dialogue.
</em></p>

---

## 🌟 Current Capabilities (2025)

### 🧠 Character Memory & Backstory
- YAML‑based long‑term memory and persona system per character.
- Stores dialogue summaries, emotional tone, and factual events.
- Compresses and sends backstory + chat context to LLM for continuity.
- Under development: **Backstory Forge UI** for interactive personality editing.

### 💬 Conversational Intelligence
- In‑character dialogue powered by either:
  - **OpenAI GPT‑4o‑mini** (default), or  
  - **Ollama local models** (e.g. `llama3.1:8b`, `phi3:3.8b`).
- Supports evolving context, internal thoughts, and expressive narrative output.
- Memory and persona data dynamically influence tone and response style.

### 🎨 Visual Generation & Editing
- Dynamic image rendering through **ComfyUI** pipelines.
- Automatic visual updates from textual prompts: pose, expression, camera, mood.
- Seamless 2‑way link between dialogue and image states.
- Local file structure supports scene‑based rendering and album history.

### 🧩 Architecture Integration
- Unified state object defines `persona`, `memory`, `scene`, and `emotion`.
- Modular backend lets components (LLM, image, memory) sync in real time.
- Supports remote or LAN‑based model execution.

---

## 🚧 In Development

| Area | Description |
|------|--------------|
| **Backstory Forge** | Web tool for creating and editing character memories, traits, and histories. |
| **Emotion Feedback** | Integrating webcam‑based user emotion detection (ML Kit / MediaPipe). |
| **Avatar Self‑Perception** | LLMs gain awareness of their own appearance via image captioning (CLIP / BLIP). |
| **Unified Scene Manager** | Links image state, conversation history, and world context. |

---

## 🛠️ Dependencies

| Component | Purpose |
|------------|----------|
| **Gradio** | Web UI and reactive state management |
| **Requests** | Communication with LLMs and ComfyUI |
| **PyYAML** | Memory persistence per character |
| **ComfyUI** | Image generation and editing workflows |
| **Ollama (optional)** | Local LLM execution |

Install with:
```bash
pip install gradio requests pyyaml pillow
```

---

## 🚀 Usage

1. Launch **ComfyUI** with your `nunchaku_edit.py` workflow.
2. Set your API key (for OpenAI users):
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
3. Start Avatarium:
   ```bash
   python companion_web_v10g.py
   ```
4. Open the app at `http://127.0.0.1:7860` and begin your conversation.
5. Watch the character’s dialogue, memory, and visuals evolve in real time.

---

## 🧭 Roadmap

| Phase | Focus | Description |
|--------|--------|-------------|
| **1. Consolidation** | Unified architecture | Merge LLM, image, and memory systems under a single state model. |
| **2. Emotion Layer** | User expression integration | Webcam emotion detection → LLM awareness. |
| **3. Self‑Perception** | Visual feedback to avatars | Let avatars “see themselves” via image analysis. |
| **4. Tooling & UX** | Creator‑facing tools | Backstory Forge, Scene Editor, persistent world logs. |

---

## 🗂️ Repository Layout

```
/companion_web_v10g.py   # Main app: LLM + ComfyUI + Gradio interface
/memory_manager.py       # Persistent per‑character memory
/album_manager.py        # Image gallery and metadata
/memory/                 # Character YAMLs
/assets/                 # Character portraits and scenes
/outputs/                # Rendered images
```

---

## 💬 Acknowledgements

Developed collaboratively with iterative AI assistance by **Stephen Hicks**.  
Part of the broader **Fusion Shift / Avatarium** universe — exploring intimacy, presence, and the boundaries between AI, art, and storytelling.

---

© 2025 Stephen Hicks — Released under the **MIT License**.
