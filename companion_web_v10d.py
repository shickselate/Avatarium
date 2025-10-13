
#!/usr/bin/env python3
"""
companion_web_v10d.py
Adds **Structured Edit Mode** for Qwen-Edit style instructions:
- Slot-based prompt with explicit sections (BACKGROUND, BODY, FACE, HANDS/PROPS, CAMERA/LIGHT, STYLE).
- Per-aspect change toggles + strength sliders to push bigger changes.
- Editable prompt template with {{placeholders}} filled by the Scene Composer.
- Variation slider to add "significant/dramatic change" cues.
Keeps: heartbeat sticky image, input source selector, chat, gating, manual render.
"""

import os, json, glob, subprocess, datetime, time, re, random
import requests
import gradio as gr

MODEL_OPENAI_DEFAULT = "gpt-4o-mini"
MODEL_OLLAMA_DEFAULT = "llama3"

DEFAULT_SYSTEM_PROMPT = """You are a warm, cinematic VR companion.
Produce a compact JSON object with:
- reply: a short friendly answer (<=2 sentences), in-character, reflecting the user's vibe.
- intent: short words that capture *place/scene/object* if any (e.g., "gothic cathedral", "rainy street").
- beats: optional bullet-ish hints for mood/action (e.g., "hushed", "hands clasped", "dust rays").

If you suggest a visual change, ALSO include a bracketed tag line on a new line:
[expression=soft_smile; head_pose=look_right; camera=static; motion=subtle_breathing]

Return ONLY JSON for the main object, then (optionally) the bracketed line.
"""

# ---- State outside Gradio ----
last_sent_plan = None
last_sent_time = 0.0

# ---------- LLM backends ----------
def llm_openai(messages, model):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": messages, "temperature": 0.8,
            "response_format": {"type": "json_object"}}
    r = requests.post(url, headers=headers, json=data, timeout=60)
    r.raise_for_status()
    j = r.json()
    content = j.get("choices",[{}])[0].get("message",{}).get("content","{}")
    return json.loads(content)

def llm_ollama(messages, model):
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    url = f"{host.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False, "format": {"type":"json_object"}}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json().get("message",{}).get("content","{}")
    return json.loads(content)

def extract_plan_bracketed(text: str):
    if not text: return {}
    if "[" in text and "]" in text:
        try:
            inner = text.split("[",1)[1].split("]",1)[0]
            parts = [p.strip() for p in inner.split(";") if p.strip()]
            out = {}
            for p in parts:
                if "=" in p:
                    k,v = p.split("=",1)
                    out[k.strip()] = v.strip()
            return out
        except Exception:
            return {}
    return {}

# ---------- Env kits & style packs ----------
ENV_KITS = [
    (r"\b(cathedral|gothic)\b", "gothic cathedral interior, ribbed vaults, pointed arches, stained glass windows, sunbeams with dust motes"),
    (r"\b(church)\b", "quiet church nave, wooden pews, candlelight, stained glass glow"),
    (r"\b(forest|woods)\b", "misty forest clearing, dappled light, mossy stones, drifting fog"),
    (r"\b(beach|shore)\b", "windswept beach at golden hour, wet sand reflections, sea spray"),
    (r"\b(street|alley)\b", "rainy city street at night, neon reflections, puddles, umbrellas"),
    (r"\b(library)\b", "old library, towering bookshelves, warm lamp light, floating dust"),
]

STYLE_PACKS = {
    "cinema": ["cinematic lighting", "volumetric rays", "filmic contrast"],
    "portrait": ["85mm lens", "shallow depth of field", "soft background bokeh"],
    "moody": ["moody grading", "subtle grain", "soft shadows"],
    "fantasy": ["ethereal glow", "ornate details", "arcane dust"],
    "": []
}

def find_env(user_text: str) -> str:
    t = (user_text or "").lower()
    for pattern, desc in ENV_KITS:
        if re.search(pattern, t):
            return desc
    return ""

# ---------- Character planning ----------
def heuristics_to_plan(text: str):
    t = (text or "").lower()
    plan = {"expression":"neutral","head_pose":"look_towards_viewer","camera":"static","motion":"subtle_breathing"}
    if any(x in t for x in ["smile","soft smile","gentle smile"]): plan["expression"]="soft_smile"
    if "sad" in t or "melancholy" in t: plan["expression"]="subtle_sad"
    if "frown" in t or "angry" in t: plan["expression"]="slight_frown"
    if "look right" in t or "to the right" in t: plan["head_pose"]="look_right"
    if "look left" in t or "to the left" in t: plan["head_pose"]="look_left"
    if "look away" in t: plan["head_pose"]="look_away"
    if "zoom" in t or "close" in t: plan["camera"]="zoom_in"
    if "wide" in t: plan["camera"]="wide_view"
    if "profile" in t or "side" in t: plan["camera"]="side_angle"
    if "over shoulder" in t or "behind" in t: plan["camera"]="behind_over_shoulder"
    if "still" in t: plan["motion"]="still"
    return plan

def call_llm(user_msg: str, history_json_pairs, system_prompt: str, provider: str, model: str):
    provider = (provider or "none").lower().strip()
    messages = [{"role": "system", "content": system_prompt}]
    for u, r in history_json_pairs[-6:]:
        messages += [{"role":"user","content":u},
                     {"role":"assistant","content":json.dumps(r)}]
    messages.append({"role":"user","content":user_msg})

    plan_json = {"reply": "I'm here—tell me more.", "intent": "", "beats": ["soft lighting"]}
    backend = "fallback"

    try:
        if provider == "openai":
            backend = "openai"
            plan_json = llm_openai(messages, model or MODEL_OPENAI_DEFAULT)
        elif provider == "ollama":
            backend = "ollama"
            plan_json = llm_ollama(messages, model or MODEL_OLLAMA_DEFAULT)
        else:
            backend = "fallback"
    except Exception as e:
        backend = f"error: {e.__class__.__name__}"

    reply = (plan_json.get("reply") or "").strip()
    intent = (plan_json.get("intent") or "").strip()
    beats = plan_json.get("beats") or []
    bracket = extract_plan_bracketed(reply)
    structured = {"expression":"neutral","head_pose":"look_towards_viewer","camera":"static","motion":"subtle_breathing"}
    structured.update(heuristics_to_plan(reply + " " + intent + " " + ", ".join(beats)))
    structured.update(bracket)
    return {"reply": reply, "intent": intent, "beats": beats}, structured, backend

# ---------- ComfyUI helpers ----------
def render_visual(visual: str, prefix: str, seed: int, input_image: str, workflow: str):
    cmd = ["python", "nunchaku_edit.py",
           "--positive", visual,
           "--prefix", prefix,
           "--seed", str(seed)]
    if input_image:
        cmd += ["--input_image", input_image]
    if workflow:
        cmd += ["--workflow", workflow]
    print(f"[render] {visual}")
    subprocess.run(cmd, check=True)

def latest_output_path(outputs_dir: str = "./outputs"):
    files = glob.glob(os.path.join(outputs_dir, "*.png"))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def upload_to_comfy(image_path: str, host="127.0.0.1", port=8188, overwrite=True) -> str:
    url = f"http://{host}:{port}/upload/image"
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/png")}
        data = {"overwrite": "true" if overwrite else "false"}
        r = requests.post(url, files=files, data=data, timeout=30)
    r.raise_for_status()
    meta = r.json()
    return meta.get("name") or os.path.basename(image_path)

# ---------- Change scoring & gating ----------
def change_score(prev: dict, curr: dict) -> float:
    if not prev: return 4.0
    keys = ["expression","head_pose","camera","motion"]
    return sum(1.0 for k in keys if prev.get(k)!=curr.get(k))

# ---------- Heartbeat ----------
def hb_tick(current_path, outputs_dir):
    newest = latest_output_path(outputs_dir) or current_path
    return newest, newest

# ---------- Structured Edit composer ----------
DEFAULT_TEMPLATE = (
"EDIT INSTRUCTIONS (clear, actionable):\n"
"BACKGROUND → {{BACKGROUND}}\n"
"FACE → {{FACE}}\n"
"BODY / GESTURE → {{BODY}}\n"
"HANDS / PROPS → {{HANDS}}\n"
"CAMERA / LIGHT → {{CAMERA_LIGHT}}\n"
"STYLE → {{STYLE}}\n"
"\n"
"Make the above changes visibly and consistently.\n"
"Avoid keeping the previous background or identical pose. Keep identity consistent."
)

def build_sections(structured, env_hint, beats, user_text, style, boost, vary):
    # Environment
    env = env_hint or find_env(user_text) or "clean studio backdrop"
    if vary > 1.5:
        env = "dramatically different " + env
    elif vary > 0.9:
        env = "significantly different " + env
    # Face
    expr_map = {"soft_smile":"soft smile","subtle_sad":"subtle sadness","slight_frown":"slight frown","neutral":"neutral expression"}
    gaze_map = {"look_towards_viewer":"eyes to camera","look_right":"gaze right","look_left":"gaze left","look_away":"looking away"}
    face = f"{expr_map.get(structured.get('expression','neutral'),'neutral')}, {gaze_map.get(structured.get('head_pose','look_towards_viewer'),'eyes to camera')}"
    if boost > 1.2: face += ", refined micro-expression"
    # Body
    cam_map = {"zoom_in":"close-up", "zoom_out":"wide view","side_angle":"side angle","behind_over_shoulder":"over-shoulder angle","wide_view":"wide view","static":"eye-level angle"}
    body = "relaxed posture"
    if "side angle" in cam_map.get(structured.get("camera","static"),""):
        body += ", slight turn of shoulders"
    if boost > 1.2:
        body += ", natural breathing presence"
    # Hands/Props (infer from beats/user text)
    hands = ""
    t = (user_text or "").lower()
    if "candle" in t or "cathedral" in t: hands = "hold a small candle with warm flame"
    if "book" in t or "library" in t: hands = "gently hold an old book"
    if not hands and beats: hands = ", ".join(beats)
    if not hands: hands = "hands relaxed, visible"
    # Camera/Light
    cam = cam_map.get(structured.get("camera","static"), "eye-level angle")
    cl = [cam, "cinematic lighting"]
    if style == "portrait": cl += ["85mm lens", "shallow depth of field"]
    if style == "cinema": cl += ["volumetric rays"]
    if style == "moody": cl += ["moody grading"]
    if style == "fantasy": cl += ["ethereal glow"]
    if boost > 1.2: cl += ["rich texture detail", "depth layering"]
    camera_light = ", ".join(cl)
    # Style
    style_bits = STYLE_PACKS.get(style, [])
    if boost > 1.2: style_bits += ["rim light highlights"]
    style_line = ", ".join(style_bits) if style_bits else "natural look"
    return {
        "BACKGROUND": env,
        "FACE": face,
        "BODY": body,
        "HANDS": hands,
        "CAMERA_LIGHT": camera_light,
        "STYLE": style_line
    }

def compose_structured_prompt(sections, template, use_background, use_body, use_face, use_hands, use_camera, use_style,
                              s_bg, s_body, s_face, s_hands, s_camera, s_style):
    # Inject intensity adverbs
    def intensify(text, s):
        if s >= 1.8: prefix = "dramatically "
        elif s >= 1.2: prefix = "significantly "
        elif s >= 0.8: prefix = "clearly "
        else: prefix = ""
        return (prefix + text) if text else text

    payload = dict(sections)
    if not use_background: payload["BACKGROUND"] = "keep current background"
    else: payload["BACKGROUND"] = intensify(payload["BACKGROUND"], s_bg)

    if not use_face: payload["FACE"] = "keep face"
    else: payload["FACE"] = intensify(payload["FACE"], s_face)

    if not use_body: payload["BODY"] = "keep body pose"
    else: payload["BODY"] = intensify(payload["BODY"], s_body)

    if not use_hands: payload["HANDS"] = "hands unchanged"
    else: payload["HANDS"] = intensify(payload["HANDS"], s_hands)

    if not use_camera: payload["CAMERA_LIGHT"] = "keep framing and lighting"
    else: payload["CAMERA_LIGHT"] = intensify(payload["CAMERA_LIGHT"], s_camera)

    if not use_style: payload["STYLE"] = "consistent"
    else: payload["STYLE"] = intensify(payload["STYLE"], s_style)

    out = template
    for k, v in payload.items():
        out = out.replace("{{"+k+"}}", v)
    return out

# ---------- Steps & Heartbeat ----------
def prime_step(user_msg, chat_messages, last_path):
    if not user_msg.strip():
        return "", chat_messages, chat_messages, last_path, "LLM: idle"
    chat_messages = chat_messages + [{"role":"user","content":user_msg}]
    return "", chat_messages, chat_messages, last_path, "LLM: pending"

def plan_step(chat_messages, json_history, seed,
              input_src_mode, base_image_path, custom_image_path,
              workflow, outputs_dir,
              system_prompt, continuity, last_path, provider, model,
              auto_send, threshold, cooldown_s, llmstatus, prompt_log,
              creative_boost, style_preset, backstory, baseline_mood,
              structured_mode, template_text,
              use_bg, use_body, use_face, use_hands, use_cam, use_style,
              s_bg, s_body, s_face, s_hands, s_cam, s_style, variation):
    global last_sent_plan, last_sent_time

    # last user
    user_msg = ""
    for m in reversed(chat_messages):
        if m.get("role") == "user":
            user_msg = m.get("content","")
            break

    # LLM call
    plan_json, structured, backend = call_llm(
        user_msg + ("\nBackstory: " + backstory if backstory else "") + ("\nMood: " + baseline_mood if baseline_mood else ""),
        json_history, system_prompt, provider, model
    )
    reply, intent, beats = plan_json["reply"], plan_json.get("intent",""), plan_json.get("beats",[])
    chat_messages = chat_messages + [{"role":"assistant","content":reply}]
    new_json_hist = json_history + [(user_msg, plan_json)]
    llmstatus = f"LLM: {backend}"

    # input source
    used_input = base_image_path
    if input_src_mode == "Base image":
        used_input = base_image_path
    elif input_src_mode == "Last render":
        used_input = last_path or base_image_path
        if continuity and last_path:
            try:
                used_input = upload_to_comfy(last_path)
            except Exception as e:
                print("[!] Upload failed, falling back:", e)
                used_input = last_path
    elif input_src_mode == "Custom":
        used_input = custom_image_path or base_image_path

    # Compose visual
    sections = build_sections(structured, intent, beats, user_msg, style_preset, creative_boost, variation)
    if structured_mode:
        visual = compose_structured_prompt(sections, template_text,
                                           use_bg, use_body, use_face, use_hands, use_cam, use_style,
                                           s_bg, s_body, s_face, s_hands, s_cam, s_style)
    else:
        # fallback simple one-line prompt
        visual = ", ".join([sections["BACKGROUND"], sections["FACE"], sections["BODY"], sections["HANDS"], sections["CAMERA_LIGHT"], sections["STYLE"]])

    job = {
        "visual": visual,
        "seed": int(seed),
        "base_img": used_input,
        "workflow": workflow,
        "outputs_dir": outputs_dir,
        "prefix": "chat_" + datetime.datetime.now().strftime("%H%M%S"),
        "structured": structured,
        "used_input": used_input,
    }

    # Auto-send gating
    delta = change_score(last_sent_plan, structured)
    should_send = auto_send and (delta >= float(threshold)) and ((time.time() - last_sent_time) >= float(cooldown_s))

    # log
    prompt_log = (prompt_log or []) + [f"{datetime.datetime.now().strftime('%H:%M:%S')} | {visual}"]
    prompt_log = prompt_log[-12:]

    return (chat_messages, chat_messages, new_json_hist, last_path, job,
            json.dumps(structured, indent=2), visual, llmstatus,
            f"Δ={delta:.1f} (th={threshold}, cd={cooldown_s}s)",
            should_send, prompt_log, f"Input used: {used_input}")

def render_step(job, last_path, gate_should_send, force=False):
    global last_sent_plan, last_sent_time
    if not job:
        return last_path, "No job."
    if not gate_should_send and not force:
        return last_path, "Hold: gate not passed (use Manual Render to force)."
    try:
        render_visual(job["visual"], job["prefix"], job["seed"], job["base_img"], job["workflow"])
        last_sent_plan = job.get("structured") or last_sent_plan
        last_sent_time = time.time()
        return last_path, "Rendering finished — heartbeat will swap image when saved."
    except Exception as e:
        return last_path, f"Render failed: {e}"

def force_render(job, last_path):
    return render_step(job, last_path, gate_should_send=True, force=True)

def sync_last_path(img_path):
    return img_path

def hb_tick(current_path, outputs_dir):
    newest = latest_output_path(outputs_dir) or current_path
    return newest, newest

# ---------- UI ----------
def build_app():
    with gr.Blocks(title="Companion v10d") as demo:
        gr.Markdown("## Companion v10d — Structured Edit Mode for bigger, clearer changes")

        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(label="Latest frame", type="filepath")
                used_input_label = gr.Markdown("Input used: -")
                last_visual = gr.Textbox(label="Visual prompt sent to Comfy", interactive=False, lines=6)
                prompt_log = gr.Textbox(label="Recent visual prompts", interactive=False, lines=8)
            with gr.Column(scale=1):
                chat = gr.Chatbot(type="messages", height=520)
                txt = gr.Textbox(placeholder="Tell me a scene or change (e.g., 'gothic cathedral, candle in hand')", show_label=False)

                with gr.Accordion("Settings", open=False):
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    input_src_mode = gr.Radio(choices=["Base image","Last render","Custom"], value="Base image", label="Input source")
                    base_image_path = gr.Textbox(value="ref_base.png", label="Base image (relative to ComfyUI/input/)")
                    custom_image_path = gr.Textbox(value="", label="Custom input image (optional)")
                    workflow = gr.Textbox(value="Nunchaku Qwen Edit Lightning 2509 - One Image Edit.json",
                                          label="Workflow JSON (API format)")
                    outputs_dir = gr.Textbox(value="./outputs", label="Outputs dir (downloaded by comfy_client)")
                    continuity = gr.Checkbox(value=True, label="Upload last render to Comfy input (when using 'Last render')")

                with gr.Accordion("Character & LLM", open=False):
                    sys_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, lines=10, max_lines=18, label="System prompt")
                    provider = gr.Dropdown(choices=["none","openai","ollama"], value="openai", label="LLM provider")
                    model = gr.Textbox(value=MODEL_OPENAI_DEFAULT, label="Model")
                    llmstatus = gr.Markdown("LLM: not checked")

                with gr.Accordion("Scene Composer", open=True):
                    creative_boost = gr.Slider(0.0, 2.0, value=1.2, step=0.1, label="Creative boost")
                    style_preset = gr.Dropdown(choices=["cinema","portrait","moody","fantasy",""], value="cinema", label="Style preset")
                    backstory = gr.Textbox(value="", lines=3, label="Character backstory (optional)")
                    baseline_mood = gr.Textbox(value="calm, attentive", label="Baseline mood words")
                    variation = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Background variation strength")

                with gr.Accordion("Structured Edit Mode", open=True):
                    structured_mode = gr.Checkbox(value=True, label="Use structured edit format")
                    template_text = gr.Textbox(value=DEFAULT_TEMPLATE, lines=10, label="Prompt template (editable)")

                    with gr.Row():
                        use_bg = gr.Checkbox(value=True, label="Change Background")
                        use_body = gr.Checkbox(value=True, label="Change Body/Gesture")
                        use_face = gr.Checkbox(value=True, label="Change Face")
                    with gr.Row():
                        use_hands = gr.Checkbox(value=True, label="Change Hands/Props")
                        use_cam = gr.Checkbox(value=True, label="Change Camera/Light")
                        use_style = gr.Checkbox(value=True, label="Change Style")

                    gr.Markdown("**Change intensities (per aspect)**")
                    s_bg = gr.Slider(0.0, 2.0, value=1.2, step=0.1, label="Background")
                    s_body = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Body")
                    s_face = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Face")
                    s_hands = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Hands/Props")
                    s_cam = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Camera/Light")
                    s_style = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Style")

                with gr.Accordion("Planner & Gating", open=False):
                    plan_json_box = gr.Textbox(label="Structured plan (JSON)", interactive=True, lines=8)
                    threshold = gr.Slider(0.0, 4.0, value=1.0, step=0.5, label="Auto-send change threshold (0–4)")
                    cooldown = gr.Slider(0.0, 30.0, value=3.0, step=1.0, label="Min seconds between sends")
                    auto_send = gr.Checkbox(value=True, label="Auto-send when gate passes")
                    delta_label = gr.Markdown("Δ=0.0")

                    manual_btn = gr.Button("Render now (Manual)", variant="primary")

        # States
        chat_state = gr.State([])
        json_state = gr.State([])
        job_state  = gr.State(None)
        last_path  = gr.State(None)
        gate_state = gr.State(False)
        prompt_state = gr.State([])

        # Heartbeat
        timer = gr.Timer(1.0)
        timer.tick(hb_tick, inputs=[last_path, outputs_dir], outputs=[image, last_path])

        # Step 1
        chain = txt.submit(
            prime_step,
            inputs=[txt, chat_state, last_path],
            outputs=[txt, chat_state, chat, image, llmstatus]
        )

        # Step 2
        chain = chain.then(
            plan_step,
            inputs=[chat_state, json_state, seed,
                    input_src_mode, base_image_path, custom_image_path,
                    workflow, outputs_dir,
                    sys_prompt, continuity, last_path, provider, model,
                    auto_send, threshold, cooldown, llmstatus, prompt_state,
                    creative_boost, style_preset, backstory, baseline_mood,
                    structured_mode, template_text,
                    use_bg, use_body, use_face, use_hands, use_cam, use_style,
                    s_bg, s_body, s_face, s_hands, s_cam, s_style, variation],
            outputs=[chat_state, chat, json_state, image, job_state,
                     plan_json_box, last_visual, llmstatus,
                     delta_label, gate_state, prompt_state, used_input_label]
        )

        # Step 3
        chain = chain.then(
            render_step,
            inputs=[job_state, last_path, gate_state],
            outputs=[image, delta_label]
        )

        manual_btn.click(
            force_render,
            inputs=[job_state, last_path],
            outputs=[image, delta_label]
        )

        image.change(sync_last_path, inputs=[image], outputs=[last_path])

        def ping(sys_prompt, prov, mdl):
            try:
                msg = "Say 'ready' and give an intent + beats as JSON."
                plan_json, structured, backend = call_llm(msg, [], sys_prompt, prov, mdl)
                return f"LLM: {backend} → ok"
            except Exception as e:
                return f"LLM: error: {e}"
        gr.Button("Ping LLM").click(ping, inputs=[sys_prompt, provider, model], outputs=[llmstatus])

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch()
