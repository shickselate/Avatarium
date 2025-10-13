
#!/usr/bin/env python3
"""
companion_web_v10f.py
- Default LLM provider: OpenAI (model: gpt-4o-mini)
- Stronger head/face/gesture variation controls:
  * Sliders: Head movement, Expression variance, Gesture energy
  * Structured template adds HEAD/GAZE and HANDS/GESTURES/PROPS
  * Composer encourages head tilt, look directions, talking-hands gestures
- Keeps: memory + album integration (from v10e), heartbeat, structured edit mode.
"""

import os, json, glob, subprocess, datetime, time, re, random
import requests
import gradio as gr

# Optional deps
try:
    import memory_manager as mm
except Exception as _e:
    mm = None
try:
    from album_manager import save_to_album, list_album
except Exception as _e2:
    save_to_album = None
    list_album = None

MODEL_OPENAI_DEFAULT = "gpt-4o-mini"
MODEL_OLLAMA_DEFAULT = "llama3.1:8b"

DEFAULT_SYSTEM_PROMPT_BASE = """You are a warm, cinematic VR companion.
Use the provided Character Profile to stay consistent in tone and visuals.
Return a compact JSON object with:
- reply: a short friendly answer (<=2 sentences), in-character.
- intent: short words that capture place/scene/objects if any (e.g., "gothic cathedral", "rainy street").
- beats: optional bullet-ish hints for mood/action (e.g., "hushed", "hands clasped").

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
    try:
        return json.loads(content)
    except Exception:
        alt = {"model": model, "messages": messages + [{"role":"system","content":"Return ONLY a JSON object."}], "stream": False}
        r2 = requests.post(url, json=alt, timeout=60)
        r2.raise_for_status()
        content2 = r2.json().get("message",{}).get("content","{}")
        return json.loads(content2)

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
def heuristics_to_plan(text: str, head_variance: float, expr_variance: float, gesture_energy: float):
    t = (text or "").lower()
    plan = {"expression":"neutral","head_pose":"look_towards_viewer","camera":"static","motion":"subtle_breathing"}

    # Base extraction
    if any(x in t for x in ["smile","soft smile","gentle smile"]): plan["expression"]="soft_smile"
    if "sad" in t or "melancholy" in t: plan["expression"]="subtle_sad"
    if "frown" in t or "angry" in t: plan["expression"]="slight_frown"
    if "look right" in t or "to the right" in t: plan["head_pose"]="look_right"
    if "look left" in t or "to the left" in t: plan["head_pose"]="look_left"
    if "look up" in t: plan["head_pose"]="look_up"
    if "look down" in t: plan["head_pose"]="look_down"
    if "zoom" in t or "close" in t: plan["camera"]="zoom_in"
    if "wide" in t: plan["camera"]="wide_view"
    if "profile" in t or "side" in t: plan["camera"]="side_angle"
    if "over shoulder" in t or "behind" in t: plan["camera"]="behind_over_shoulder"
    if "still" in t: plan["motion"]="still"

    # Controlled randomness to encourage movement/expressivity
    head_options = ["look_right","look_left","look_up","look_down","look_away","look_towards_viewer"]
    if random.random() < min(1.0, head_variance * 0.5):
        plan["head_pose"] = random.choice(head_options)

    expr_options = ["soft_smile","subtle_sad","slight_frown","neutral","soft_smile"]
    if random.random() < min(1.0, expr_variance * 0.5):
        plan["expression"] = random.choice(expr_options)

    cam_options = ["static","zoom_in","wide_view","side_angle","behind_over_shoulder"]
    if random.random() < 0.25 * head_variance:
        plan["camera"] = random.choice(cam_options)

    if gesture_energy > 1.2 and plan.get("motion","") != "still":
        plan["motion"] = "subtle_breathing"

    return plan

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
"FACE (expression) → {{FACE}}\n"
"HEAD / GAZE → {{HEAD}}\n"
"BODY / GESTURE → {{BODY}}\n"
"HANDS / GESTURES / PROPS → {{HANDS}}\n"
"WARDROBE → {{WARDROBE}}\n"
"CAMERA / LIGHT → {{CAMERA_LIGHT}}\n"
"STYLE → {{STYLE}}\n"
"\n"
"Make the above changes visibly and consistently.\n"
"Avoid keeping the previous background or identical pose. Keep identity consistent."
)

def wardrobe_from_context(user_text: str, intent: str, style: str):
    t = (user_text + " " + (intent or "")).lower()
    if "cathedral" in t or "church" in t:
        return "dark wool coat with high collar, subtle embroidery, modest silhouette"
    if "library" in t:
        return "soft knit cardigan over simple blouse, muted earth tones"
    if "beach" in t or "shore" in t:
        return "light linen dress, wind-kissed fabric, natural tones"
    if "street" in t or "alley" in t:
        return "long trench coat, layered scarf, urban chic"
    if "forest" in t or "woods" in t:
        return "weathered cloak, leather belt, textured fabrics"
    if style == "fantasy":
        return "ornate cloak, subtle metallic trim, storybook details"
    if style == "moody":
        return "matte black layers, understated tailoring"
    if style == "portrait":
        return "clean monochrome top, minimal patterns"
    return "simple, timeless attire"

def build_sections(structured, env_hint, beats, user_text, style, boost, vary,
                   head_variance, expr_variance, gesture_energy):
    # Environment
    env = env_hint or find_env(user_text) or "clean studio backdrop"
    if vary > 1.5:
        env = "dramatically different " + env
    elif vary > 0.9:
        env = "significantly different " + env

    # Face (expression)
    expr_map = {
        "soft_smile":"soft smile",
        "subtle_sad":"subtle sadness",
        "slight_frown":"slight frown",
        "neutral":"neutral expression"
    }
    face = expr_map.get(structured.get('expression','neutral'),'neutral expression')
    if expr_variance > 1.2: face += ", refined micro-expression"

    # Head / gaze
    gaze_map = {
        "look_towards_viewer":"eyes to camera",
        "look_right":"gaze right",
        "look_left":"gaze left",
        "look_away":"looking away",
        "look_up":"chin slightly up, gaze upward",
        "look_down":"chin slightly down, gaze downward"
    }
    head = gaze_map.get(structured.get('head_pose','look_towards_viewer'),'eyes to camera')
    if head_variance > 1.2:
        head += ", gentle head tilt"

    # Body / gesture (torso posture)
    body = "relaxed posture"
    if boost > 1.0:
        body += ", natural breathing presence"

    # Hands / gestures / props
    t = (user_text or "").lower()
    hands = ""
    if "candle" in t or "cathedral" in t: hands = "talking with one hand while holding a small candle"
    if "book" in t or "library" in t: hands = "gently hold an old book, one hand gesturing softly"
    if not hands and beats:
        hands = ", ".join(beats)
    if not hands:
        if gesture_energy >= 1.6:
            hands = "expressive open-palm gestures near chest height"
        elif gesture_energy >= 1.1:
            hands = "subtle talking-hands gestures, one hand slightly raised"
        else:
            hands = "hands relaxed, lightly animated"

    # Wardrobe
    wardrobe = wardrobe_from_context(user_text, env_hint or "", style)
    if boost > 1.2: wardrobe += ", elegant fabric texture"

    # Camera/Light
    cam_map = {"static":"eye-level angle","zoom_in":"close-up","zoom_out":"wide view","wide_view":"wide view","side_angle":"side angle","behind_over_shoulder":"over-shoulder angle"}
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
        "HEAD": head,
        "BODY": body,
        "HANDS": hands,
        "WARDROBE": wardrobe,
        "CAMERA_LIGHT": camera_light,
        "STYLE": style_line
    }

def compose_structured_prompt(sections, template, use_background, use_body, use_face, use_head, use_hands, use_wardrobe, use_camera, use_style,
                              s_bg, s_body, s_face, s_head, s_hands, s_wardrobe, s_camera, s_style):
    def intensify(text, s):
        if s >= 1.8: prefix = "dramatically "
        elif s >= 1.2: prefix = "significantly "
        elif s >= 0.8: prefix = "clearly "
        else: prefix = ""
        return (prefix + text) if text else text

    payload = dict(sections)
    payload["BACKGROUND"] = intensify(payload["BACKGROUND"], s_bg) if use_background else "keep current background"
    payload["FACE"] = intensify(payload["FACE"], s_face) if use_face else "keep face"
    payload["HEAD"] = intensify(payload["HEAD"], s_head) if use_head else "keep head orientation"
    payload["BODY"] = intensify(payload["BODY"], s_body) if use_body else "keep body pose"
    payload["HANDS"] = intensify(payload["HANDS"], s_hands) if use_hands else "hands unchanged"
    payload["WARDROBE"] = intensify(payload["WARDROBE"], s_wardrobe) if use_wardrobe else "keep wardrobe"
    payload["CAMERA_LIGHT"] = intensify(payload["CAMERA_LIGHT"], s_camera) if use_camera else "keep framing and lighting"
    payload["STYLE"] = intensify(payload["STYLE"], s_style) if use_style else "consistent"

    out = template
    for k, v in payload.items():
        out = out.replace("{{"+k+"}}", v)
    return out

# ---------- Memory helpers ----------
def memory_enabled():
    return mm is not None

def load_or_new_memory(char_name: str, archetype="", voice_style=""):
    path = f"memory/{char_name}.yaml"
    if not memory_enabled():
        return None, path, "Memory disabled (PyYAML or memory_manager missing)."
    try:
        mem = mm.load(path)
        if not mem.get("identity",{}).get("name"):
            mem = mm.new_character(path, name=char_name, archetype=archetype, voice_style=voice_style)
        return mem, path, f"Loaded memory for {char_name}."
    except Exception as e:
        return None, path, f"Memory error: {e}"

def memory_snapshot_block(mem):
    if not mem: return ""
    snap = mm.snapshot_for_prompt(mem)
    return json.dumps(snap, ensure_ascii=False)

# ---------- Steps & Heartbeat ----------
def prime_step(user_msg, chat_messages, last_path, mem, mem_path):
    if not user_msg.strip():
        return "", chat_messages, chat_messages, last_path, "LLM: idle", mem
    chat_messages = chat_messages + [{"role":"user","content":user_msg}]
    if mem is not None and mm is not None:
        try:
            mm.append_transcript(mem, "user", user_msg)
            mm.save(mem_path, mem)
        except Exception:
            pass
    return "", chat_messages, chat_messages, last_path, "LLM: pending", mem

def call_llm(user_msg: str, history_json_pairs, system_prompt: str, provider: str, model: str,
             head_variance: float, expr_variance: float, gesture_energy: float):
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

    structured = heuristics_to_plan(reply + " " + intent + " " + ", ".join(beats),
                                    head_variance=head_variance,
                                    expr_variance=expr_variance,
                                    gesture_energy=gesture_energy)
    structured.update(bracket)
    return {"reply": reply, "intent": intent, "beats": beats}, structured, backend

def plan_step(chat_messages, json_history, seed,
              input_src_mode, base_image_path, custom_image_path,
              workflow, outputs_dir,
              system_prompt, continuity, last_path, provider, model,
              auto_send, threshold, cooldown_s, llmstatus, prompt_log,
              creative_boost, style_preset, backstory, baseline_mood, variation,
              structured_mode, template_text,
              use_bg, use_body, use_face, use_head, use_hands, use_wardrobe, use_cam, use_style,
              s_bg, s_body, s_face, s_head, s_hands, s_wardrobe, s_cam, s_style,
              head_variance, expr_variance, gesture_energy,
              mem, mem_path, char_name):
    global last_sent_plan, last_sent_time

    user_msg = ""
    for m in reversed(chat_messages):
        if m.get("role") == "user":
            user_msg = m.get("content","")
            break

    # Build memory-aware system prompt
    sys_prompt = DEFAULT_SYSTEM_PROMPT_BASE
    if system_prompt and system_prompt.strip():
        sys_prompt = system_prompt
    if mem is not None and mm is not None:
        profile_block = memory_snapshot_block(mem)
        sys_prompt = sys_prompt + "\n\n[Character Profile JSON]\n" + profile_block

    # LLM call
    plan_json, structured, backend = call_llm(
        user_msg + ("\nBackstory: " + backstory if backstory else "") + ("\nMood: " + baseline_mood if baseline_mood else ""),
        json_history, sys_prompt, provider, model,
        head_variance, expr_variance, gesture_energy
    )
    reply, intent, beats = plan_json["reply"], plan_json.get("intent",""), plan_json.get("beats",[])
    chat_messages = chat_messages + [{"role":"assistant","content":reply}]
    new_json_hist = json_history + [(user_msg, plan_json)]
    llmstatus = f"LLM: {backend}"

    # Memory
    if mem is not None and mm is not None:
        try:
            mm.append_transcript(mem, "assistant", reply)
            mm.append_gesture(mem, structured)
            mm.save(mem_path, mem)
        except Exception:
            pass

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

    # Compose
    sections = build_sections(structured, intent, beats, user_msg, style_preset, creative_boost, variation,
                              head_variance, expr_variance, gesture_energy)
    if structured_mode:
        visual = compose_structured_prompt(sections, template_text,
                                           use_bg, use_body, use_face, use_head, use_hands, use_wardrobe, use_cam, use_style,
                                           s_bg, s_body, s_face, s_head, s_hands, s_wardrobe, s_cam, s_style)
    else:
        visual = ", ".join([sections["BACKGROUND"], sections["FACE"], sections["HEAD"], sections["BODY"],
                            sections["HANDS"], sections["WARDROBE"], sections["CAMERA_LIGHT"], sections["STYLE"]])

    job = {
        "visual": visual,
        "seed": int(seed),
        "base_img": used_input,
        "workflow": workflow,
        "outputs_dir": outputs_dir,
        "prefix": "chat_" + datetime.datetime.now().strftime("%H%M%S"),
        "structured": structured,
        "used_input": used_input,
        "char_name": char_name or "Companion"
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
            should_send, prompt_log, f"Input used: {used_input}", mem)

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

# ----- Album integration -----
def save_current_to_album(latest_image_path, composed_visual, structured_plan_json, used_input_label, char_name, mem, mem_path):
    if save_to_album is None or list_album is None:
        return "Album unavailable (album_manager missing).", []
    if not latest_image_path:
        return "No current image to save.", []
    try:
        structured_plan = {}
        if structured_plan_json:
            try:
                structured_plan = json.loads(structured_plan_json)
            except Exception:
                structured_plan = {"raw": structured_plan_json}
        meta = {
            "prompt": composed_visual,
            "plan": structured_plan,
            "used_input": used_input_label,
        }
        p = save_to_album(char_name or "Companion", latest_image_path, meta)
        if mem is not None and mm is not None:
            try:
                mm.add_episode(mem, "Saved frame", f"Saved {p.name} with prompt.")
                mm.save(mem_path, mem)
            except Exception:
                pass
        items = list_album(char_name or "Companion", limit=12)
        gallery = []
        for i in items:
            cap = (i.get("prompt") or "")[:80]
            thumb = i.get("thumb_path")
            if thumb and os.path.exists(thumb):
                gallery.append((thumb, cap))
            else:
                imgp = i.get("image_path")
                gallery.append((imgp, cap))
        return f"Saved: {p.name}", gallery
    except Exception as e:
        return f"Album save failed: {e}", []

# ----- Memory buttons -----
def new_character(name, archetype, voice_style):
    mem, path, status = load_or_new_memory(name or "Companion", archetype, voice_style)
    return mem, path, f"New character ready: {status}"

def save_memory(mem, mem_path):
    if mem is None or mm is None:
        return "Memory unavailable."
    try:
        mm.save(mem_path, mem)
        return "Memory saved."
    except Exception as e:
        return f"Memory save failed: {e}"

def pin_fact(mem, mem_path, fact):
    if mem is None or mm is None:
        return "Memory unavailable."
    try:
        mm.pin_fact(mem, fact)
        mm.save(mem_path, mem)
        return "Pinned."
    except Exception as e:
        return f"Pin failed: {e}"

def summarize_session(mem, mem_path, transcript_tail_json, provider, model):
    if mem is None or mm is None:
        return "Memory unavailable."
    try:
        tail = transcript_tail_json if isinstance(transcript_tail_json, list) else json.loads(transcript_tail_json or "[]")
    except Exception:
        tail = []
    text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in tail[-30:])
    sys = "Write a concise 2–3 sentence summary of the conversation, focusing on decisions, emotions, scene/pose changes and gestures."
    messages = [{"role":"system","content":sys},{"role":"user","content":text or "No content."}]
    try:
        if (provider or "openai").lower() == "openai":
            resp = llm_openai(messages, model or MODEL_OPENAI_DEFAULT)
        else:
            resp = llm_ollama(messages, model or MODEL_OLLAMA_DEFAULT)
        summary = resp if isinstance(resp, str) else (resp.get("reply") or resp.get("summary") or "Summary.")
    except Exception:
        summary = "Companion and user explored scenes and emotions with visible head turns, gestures, and background changes."
    try:
        mm.add_episode(mem, "Session summary", str(summary))
        mm.update_rolling_summary(mem)
        mm.save(mem_path, mem)
        return "Session summarized."
    except Exception as e:
        return f"Summarize failed: {e}"

# ---------- UI ----------
def build_app():
    with gr.Blocks(title="Companion v10f — Head/Expression/Gesture+") as demo:
        gr.Markdown("## Companion v10f — OpenAI default + enhanced head/face/gestures")

        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(label="Latest frame", type="filepath")
                used_input_label = gr.Markdown("Input used: -")
                last_visual = gr.Textbox(label="Visual prompt sent to Comfy", interactive=False, lines=6)
                prompt_log = gr.Textbox(label="Recent visual prompts", interactive=False, lines=8)
                gr.Markdown("### Album")
                album_gallery = gr.Gallery(label="Album", columns=4, height=200)
                save_album_btn = gr.Button("Save current frame to Album")

            with gr.Column(scale=1):
                chat = gr.Chatbot(type="messages", height=520)
                txt = gr.Textbox(placeholder="Scene or change (e.g., 'rainy street; look left; emphatic hands')", show_label=False)

                with gr.Accordion("Settings", open=False):
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    input_src_mode = gr.Radio(choices=["Base image","Last render","Custom"], value="Base image", label="Input source")
                    base_image_path = gr.Textbox(value="ref_base.png", label="Base image (relative to ComfyUI/input/)")
                    custom_image_path = gr.Textbox(value="", label="Custom input image (optional)")
                    workflow = gr.Textbox(value="Nunchaku Qwen Edit Lightning 2509 - One Image Edit.json",
                                          label="Workflow JSON (API format)")
                    outputs_dir = gr.Textbox(value="./outputs", label="Outputs dir (downloaded by comfy_client)")
                    continuity = gr.Checkbox(value=True, label="Upload last render to Comfy input (when using 'Last render')")

                with gr.Accordion("Character & LLM", open=True):
                    char_name = gr.Textbox(value="Ari", label="Character name")
                    char_archetype = gr.Textbox(value="Gentle guide", label="Archetype")
                    char_voice = gr.Textbox(value="warm, calm, slightly playful", label="Voice style")
                    provider = gr.Dropdown(choices=["openai","ollama","none"], value="openai", label="LLM provider")
                    model = gr.Textbox(value=MODEL_OPENAI_DEFAULT, label="Model")
                    llmstatus = gr.Markdown("LLM: not checked")

                    with gr.Row():
                        new_char_btn = gr.Button("New Character")
                        save_mem_btn = gr.Button("Save Memory")

                    with gr.Row():
                        pin_text = gr.Textbox(value="", label="Pin fact (e.g., 'User prefers moody lighting')")
                        pin_btn = gr.Button("Pin")
                    with gr.Row():
                        summarize_btn = gr.Button("Summarize Session → Memory")

                with gr.Accordion("Scene Composer", open=True):
                    creative_boost = gr.Slider(0.0, 2.0, value=1.2, step=0.1, label="Creative boost")
                    style_preset = gr.Dropdown(choices=["cinema","portrait","moody","fantasy",""], value="cinema", label="Style preset")
                    backstory = gr.Textbox(value="", lines=3, label="Character backstory (optional)")
                    baseline_mood = gr.Textbox(value="calm, attentive", label="Baseline mood words")
                    variation = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Background variation strength")

                    gr.Markdown("**Expressivity controls**")
                    head_variance = gr.Slider(0.0, 2.0, value=1.2, step=0.1, label="Head movement")
                    expr_variance = gr.Slider(0.0, 2.0, value=1.1, step=0.1, label="Expression variance")
                    gesture_energy = gr.Slider(0.0, 2.0, value=1.3, step=0.1, label="Gesture energy (hands)")

                with gr.Accordion("Structured Edit Mode", open=True):
                    structured_mode = gr.Checkbox(value=True, label="Use structured edit format")
                    template_text = gr.Textbox(value=DEFAULT_TEMPLATE, lines=14, label="Prompt template (editable)")

                    with gr.Row():
                        use_bg = gr.Checkbox(value=True, label="Change Background")
                        use_body = gr.Checkbox(value=True, label="Change Body/Gesture")
                        use_face = gr.Checkbox(value=True, label="Change Face")
                        use_head = gr.Checkbox(value=True, label="Change Head/Gaze")
                    with gr.Row():
                        use_hands = gr.Checkbox(value=True, label="Change Hands/Gestures/Props")
                        use_wardrobe = gr.Checkbox(value=True, label="Change Wardrobe")
                        use_cam = gr.Checkbox(value=True, label="Change Camera/Light")
                        use_style = gr.Checkbox(value=True, label="Change Style")

                    gr.Markdown("**Change intensities (per aspect)**")
                    s_bg = gr.Slider(0.0, 2.0, value=1.3, step=0.1, label="Background")
                    s_body = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Body")
                    s_face = gr.Slider(0.0, 2.0, value=1.2, step=0.1, label="Face")
                    s_head = gr.Slider(0.0, 2.0, value=1.4, step=0.1, label="Head/Gaze")
                    s_hands = gr.Slider(0.0, 2.0, value=1.4, step=0.1, label="Hands/Gestures/Props")
                    s_wardrobe = gr.Slider(0.0, 2.0, value=1.2, step=0.1, label="Wardrobe")
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
        mem_state = gr.State(None)
        mem_path_state = gr.State("memory/Ari.yaml")

        # Heartbeat
        timer = gr.Timer(1.0)
        timer.tick(hb_tick, inputs=[last_path, outputs_dir], outputs=[image, last_path])

        # New Character
        new_char_btn.click(
            new_character,
            inputs=[char_name, char_archetype, char_voice],
            outputs=[mem_state, mem_path_state, llmstatus]
        )

        save_mem_btn.click(save_memory, inputs=[mem_state, mem_path_state], outputs=[llmstatus])
        pin_btn.click(pin_fact, inputs=[mem_state, mem_path_state, pin_text], outputs=[llmstatus])
        summarize_btn.click(summarize_session, inputs=[mem_state, mem_path_state, chat_state, provider, model], outputs=[llmstatus])

        save_album_btn.click(
            save_current_to_album,
            inputs=[image, last_visual, plan_json_box, used_input_label, char_name, mem_state, mem_path_state],
            outputs=[llmstatus, album_gallery]
        )

        # Init memory on load
        def _init_mem(name, archetype, voice):
            mem, path, status = load_or_new_memory(name, archetype, voice)
            return mem, path, status
        demo.load(_init_mem, inputs=[char_name, char_archetype, char_voice], outputs=[mem_state, mem_path_state, llmstatus])

        # Step 1
        chain = txt.submit(
            prime_step,
            inputs=[txt, chat_state, last_path, mem_state, mem_path_state],
            outputs=[txt, chat_state, chat, image, llmstatus, mem_state]
        )

        # Step 2
        chain = chain.then(
            plan_step,
            inputs=[chat_state, json_state, seed,
                    input_src_mode, base_image_path, custom_image_path,
                    workflow, outputs_dir,
                    gr.Textbox(value=DEFAULT_SYSTEM_PROMPT_BASE, visible=False),
                    continuity, last_path, provider, model,
                    auto_send, threshold, cooldown, llmstatus, prompt_state,
                    creative_boost, style_preset, backstory, baseline_mood, variation,
                    structured_mode, template_text,
                    use_bg, use_body, use_face, use_head, use_hands, use_wardrobe, use_cam, use_style,
                    s_bg, s_body, s_face, s_head, s_hands, s_wardrobe, s_cam, s_style,
                    head_variance, expr_variance, gesture_energy,
                    mem_state, mem_path_state, char_name],
            outputs=[chat_state, chat, json_state, image, job_state,
                     plan_json_box, last_visual, llmstatus,
                     delta_label, gate_state, prompt_state, used_input_label, mem_state]
        )

        # Step 3
        chain = chain.then(
            render_step,
            inputs=[job_state, last_path, gate_state],
            outputs=[image, delta_label]
        )

        manual_btn.click(force_render, inputs=[job_state, last_path], outputs=[image, delta_label])
        image.change(sync_last_path, inputs=[image], outputs=[last_path])

        def ping(sys_prompt, prov, mdl):
            try:
                msg = "Say 'ready' and give an intent + beats as JSON."
                plan_json, structured, backend = call_llm(msg, [], sys_prompt, prov, mdl, 1.2, 1.1, 1.3)
                return f"LLM: {backend} → ok"
            except Exception as e:
                return f"LLM: error: {e}"
        gr.Button("Ping LLM").click(ping, inputs=[gr.Textbox(value=DEFAULT_SYSTEM_PROMPT_BASE, visible=False), provider, model], outputs=[llmstatus])

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch()
