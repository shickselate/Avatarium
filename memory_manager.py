
# memory_manager.py — minimal dependency memory layer (requires PyYAML)
# pip install pyyaml
from __future__ import annotations
import yaml, time, copy
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

DEFAULT = {
    "meta": {"version": 1, "last_updated": None},
    "identity": {"name":"Ari","pronouns":"she/they","archetype":"Gentle guide","voice_style":"warm, calm, slightly playful"},
    "backstory": {"summary":"", "key_events": [], "relationships": []},
    "motivations": {"goals": [], "fears": [], "red_lines": []},
    "traits": {"stable": [], "drifting": []},
    "visual_identity": {"anchor_details": {}, "wardrobe_presets": {}},
    "style_preferences": {"cinematography": [], "editing_bias": [], "negative_prompts": []},
    "episodic": [],
    "transcript": [],
    "gestures_log": [],
    "rolling_summary": "",
    "pinned_facts": []
}

def load(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    merged = copy.deepcopy(DEFAULT)
    # shallow merge keys
    for k, v in (data or {}).items():
        merged[k] = v
    return merged

def save(path: str, data: Dict[str, Any]):
    data = copy.deepcopy(data)
    data.setdefault("meta", {})["last_updated"] = _now_iso()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

def new_character(path: str, name: str, archetype: str = "", voice_style: str = ""):
    mem = copy.deepcopy(DEFAULT)
    mem["identity"]["name"] = name
    if archetype: mem["identity"]["archetype"] = archetype
    if voice_style: mem["identity"]["voice_style"] = voice_style
    save(path, mem)
    return mem

def append_transcript(mem: Dict[str, Any], role: str, text: str):
    mem["transcript"].append({"ts": _now_iso(), "role": role, "text": text})
    # Bound transcript length (keep last ~250 messages)
    if len(mem["transcript"]) > 250:
        mem["transcript"] = mem["transcript"][-250:]

def append_gesture(mem: Dict[str, Any], plan: Dict[str, Any]):
    mem["gestures_log"].append({"ts": _now_iso(), "plan": plan})
    if len(mem["gestures_log"]) > 200:
        mem["gestures_log"] = mem["gestures_log"][-200:]

def pin_fact(mem: Dict[str, Any], fact: str):
    if fact and fact not in mem["pinned_facts"]:
        mem["pinned_facts"].append(fact)

def add_episode(mem: Dict[str, Any], title: str, summary: str):
    mem["episodic"].append({"ts": _now_iso(), "title": title, "summary": summary})
    if len(mem["episodic"]) > 60:
        mem["episodic"] = mem["episodic"][-60:]

def update_rolling_summary(mem: Dict[str, Any]):
    # Tiny heuristic summary from identity, last 3 episodes, and top pinned facts
    name = mem.get("identity",{}).get("name","Companion")
    goals = ", ".join(mem.get("motivations",{}).get("goals", [])[:3])
    traits = ", ".join(mem.get("traits",{}).get("stable", [])[:3])
    episodes = mem.get("episodic", [])[-3:]
    epi_text = " | ".join(e.get("summary","") for e in episodes if e.get("summary"))
    facts = "; ".join(mem.get("pinned_facts", [])[:3])
    mem["rolling_summary"] = f"{name}: stable traits [{traits}]; goals [{goals}]. Recent: {epi_text}. Facts: {facts}"

def snapshot_for_prompt(mem: Dict[str, Any]) -> Dict[str, Any]:
    # Return a compact slice to inject into the LLM system prompt
    return {
        "identity": mem.get("identity",{}),
        "backstory": mem.get("backstory",{}),
        "motivations": mem.get("motivations",{}),
        "traits": mem.get("traits",{}),
        "visual_identity": mem.get("visual_identity",{}),
        "style_preferences": mem.get("style_preferences",{}),
        "pinned_facts": mem.get("pinned_facts",[]),
        "rolling_summary": mem.get("rolling_summary",""),
        "recent_episodes": mem.get("episodic", [])[-5:],
        "recent_gestures": mem.get("gestures_log", [])[-10:],
    }
