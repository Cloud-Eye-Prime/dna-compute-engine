"""
llm_bridge.py — Provider-agnostic LLM interface.
Supports: Ollama, any OpenAI-compatible endpoint, Anthropic, LXR-5 Dragon MoA, HuggingFace.
"""
import os, re, requests, yaml

SYSTEM_PROMPT = (
    "You are a Blender Python expert (bpy API 4.x).\n"
    "Output ONLY valid Python code — no markdown, no backticks, no explanations.\n"
    "The code runs inside `blender --background` and must:\n"
    "  1. import bpy\n"
    "  2. Clear the default scene.\n"
    "  3. Build the described 3D scene using bpy.ops and bpy.data.\n"
    "  4. Configure render settings via bpy.context.scene.render.\n"
    "  5. Call bpy.ops.render.render(write_still=True) at the end."
)


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    raw = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), raw)
    return yaml.safe_load(raw)


class LLMBridge:
    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)
        self.provider = self.cfg["active"]
        self.pcfg = self.cfg["providers"][self.provider]

    def set_provider(self, name):
        self.provider = name
        self.pcfg = self.cfg["providers"][name]

    def ask(self, prompt: str, model: str = None) -> str:
        dispatch = {
            "ollama":       self._ollama,
            "openai_compat": self._openai_compat,
            "anthropic":    self._anthropic,
            "lxr5":         self._lxr5,
            "hf":           self._hf,
        }
        fn = dispatch.get(self.provider)
        if not fn:
            raise ValueError(f"Unknown provider: {self.provider}")
        return fn(prompt, model)

    # ── Ollama ─────────────────────────────────────────────────────────────
    def _ollama(self, prompt, model):
        m = model or self.pcfg["model"]
        url = self.pcfg["base_url"].rstrip("/") + "/api/chat"
        r = requests.post(url, json={
            "model": m,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            "stream": False
        }, timeout=self.pcfg["timeout"])
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    # ── OpenAI-compatible (LM Studio, llama.cpp, OpenRouter, vLLM…) ───────
    def _openai_compat(self, prompt, model):
        m = model or self.pcfg["model"]
        url = self.pcfg["base_url"].rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        key = self.pcfg.get("api_key", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = requests.post(url, headers=headers, json={
            "model": m,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 4096
        }, timeout=self.pcfg["timeout"])
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # ── Anthropic ──────────────────────────────────────────────────────────
    def _anthropic(self, prompt, model):
        m = model or self.pcfg["model"]
        r = requests.post("https://api.anthropic.com/v1/messages", headers={
            "x-api-key": self.pcfg["api_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }, json={
            "model": m,
            "max_tokens": self.pcfg.get("max_tokens", 4096),
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}]
        }, timeout=self.pcfg["timeout"])
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()

    # ── LXR-5 Dragon MoA ───────────────────────────────────────────────────
    def _lxr5(self, prompt, model):
        url = self.pcfg["base_url"].rstrip("/") + "/wujiallychat"
        r = requests.post(url, headers={
            "Authorization": f"Bearer {self.pcfg['api_key']}",
            "Content-Type": "application/json"
        }, json={"message": f"{SYSTEM_PROMPT}\n\n{prompt}"},
           timeout=self.pcfg["timeout"])
        r.raise_for_status()
        return r.json().get("response", "").strip()

    # ── HuggingFace Inference API ──────────────────────────────────────────
    def _hf(self, prompt, model):
        m = model or self.pcfg["model"]
        url = self.pcfg["base_url"].rstrip("/") + "/chat/completions"
        r = requests.post(url, headers={
            "Authorization": f"Bearer {self.pcfg['api_key']}",
            "Content-Type": "application/json"
        }, json={
            "model": m,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            "max_tokens": 4096
        }, timeout=self.pcfg["timeout"])
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # ── Utility ────────────────────────────────────────────────────────────
    def generate_bpy_code(self, scene_prompt: str, output_path: str, model: str = None) -> str:
        full_prompt = (
            f"Scene description: {scene_prompt}\n\n"
            f"Set bpy.context.scene.render.filepath = \"{output_path}\"\n"
            "Include this filepath assignment in the output code."
        )
        code = self.ask(full_prompt, model)
        code = re.sub(r"^```[\w]*\n?", "", code, flags=re.MULTILINE)
        code = re.sub(r"```\s*$", "", code, flags=re.MULTILINE)
        return code.strip()
