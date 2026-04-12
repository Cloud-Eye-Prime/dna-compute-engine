"""
physics_matrix.py — LLM generates a parameter matrix for physics simulation.

The LLM returns:
  - base_scene_code:  bpy Python that builds the scene geometry
  - parameter_matrix: N variants, each with physics param overrides as bpy code

Usage:
    from physics_matrix import PhysicsMatrix
    pm = PhysicsMatrix("config.yaml")
    matrix = pm.generate("cloth sphere colliding with rigid floor", 
                         physics_type="cloth", n_variants=6)
    matrix.save("my_sim_matrix.json")
"""
import re, json, pathlib
from llm_bridge import LLMBridge, load_config

PHYSICS_TYPES = {
    "rigid_body": {
        "desc": "Rigid body dynamics: collisions, bouncing, stacking.",
        "params": ["friction", "restitution", "mass", "linear_damping", 
                   "angular_damping", "collision_shape", "substeps_per_frame"],
        "bpy_hints": "bpy.ops.rigidbody.object_add(); obj.rigid_body.friction = ..."
    },
    "soft_body": {
        "desc": "Soft/elastic body deformation.",
        "params": ["goal_stiffness", "goal_damping", "pull", "push", 
                   "bending", "mass", "gravity"],
        "bpy_hints": "obj.modifiers.new('Softbody','SOFT_BODY'); obj.soft_body.goal_stiffness = ..."
    },
    "cloth": {
        "desc": "Cloth simulation: fabric, flags, draping.",
        "params": ["tension_stiffness", "compression_stiffness", "bending_stiffness",
                   "mass", "air_damping", "quality_steps", "gravity"],
        "bpy_hints": "obj.modifiers.new('Cloth','CLOTH'); obj.modifiers['Cloth'].settings.tension_stiffness = ..."
    },
    "fluid": {
        "desc": "Mantaflow fluid (liquid or smoke).",
        "params": ["resolution_max", "viscosity_base", "viscosity_exponent",
                   "surface_tension", "use_spray", "use_foam", "timesteps_max"],
        "bpy_hints": "obj.modifiers.new('Fluid','FLUID'); obj.modifiers['Fluid'].fluid_type = 'DOMAIN'; ..."
    },
    "particles": {
        "desc": "Particle system: rain, sparks, hair, debris.",
        "params": ["count", "lifetime", "mass", "size", "factor_gravity",
                   "normal_factor", "angular_velocity_factor", "brownian_factor"],
        "bpy_hints": "obj.modifiers.new('ParticleSystem','PARTICLE_SYSTEM'); ps = obj.particle_systems[0].settings; ps.count = ..."
    },
    "geometry_nodes": {
        "desc": "Custom physics via Geometry Nodes Simulation Zones (Blender 4.x+).",
        "params": ["simulation_substeps", "force_scale", "damping", "stiffness",
                   "collision_radius", "custom_field_strength", "noise_scale"],
        "bpy_hints": "mod = obj.modifiers.new('GeoSim','NODES'); mod.node_group = bpy.data.node_groups['PhysicsSim']; mod['Input_1'] = ..."
    }
}

MATRIX_SYSTEM_PROMPT = """You are a Blender Python physics expert. Output ONLY valid JSON — no markdown, no backticks.

You will generate a physics parameter matrix: one base scene + N parameter variants.
Each variant explores a different quality or behavior of the physics simulation.

Output this exact JSON structure:
{
  "physics_type": "<type>",
  "base_scene_code": "<Python bpy code string — creates geometry, lighting, camera>",
  "parameter_matrix": [
    {
      "variant_id": "v001",
      "quality_label": "<descriptive label>",
      "description": "<what makes this variant different>",
      "param_values": { "<param_name>": <value>, ... },
      "override_code": "<Python bpy code — applies params to the scene; runs AFTER base_scene_code>"
    }
  ]
}

Rules for base_scene_code:
- import bpy at top
- Clear default scene
- Create geometry, materials, lights, camera
- Do NOT set physics yet (that goes in override_code)
- End with: bpy.context.scene.frame_start=1; bpy.context.scene.frame_end=__FRAME_END__

Rules for override_code per variant:
- Apply physics modifier + settings for THIS variant's params
- Call bpy.ops.ptcache.bake_all(bake=True) to bake the simulation
- Set filepath: bpy.context.scene.render.filepath = "__OUTPUT_PATH__"
- Call bpy.ops.render.render(animation=True) at the end
"""

class PhysicsMatrix:
    def __init__(self, config_path="config.yaml"):
        self.bridge = LLMBridge(config_path)
        self.cfg = load_config(config_path)

    def generate(self, description: str, physics_type: str = "rigid_body",
                 n_variants: int = 4, frame_end: int = 72,
                 model: str = None) -> "MatrixResult":
        ptype = PHYSICS_TYPES.get(physics_type, PHYSICS_TYPES["rigid_body"])
        user_prompt = (
            f"Scene: {description}\n"
            f"Physics type: {physics_type} — {ptype['desc']}\n"
            f"Available parameters: {', '.join(ptype['params'])}\n"
            f"bpy API hints: {ptype['bpy_hints']}\n"
            f"Generate {n_variants} variants that explore different qualities.\n"
            f"Vary parameters meaningfully: draft/preview/standard/cinematic quality,\n"
            f"or explore physical extremes (stiff/soft, heavy/light, etc.)\n"
            f"frame_end = {frame_end}"
        )
        # Set the matrix-specific system prompt
        import llm_bridge as _lb
        _old_sp = _lb.SYSTEM_PROMPT
        _lb.SYSTEM_PROMPT = MATRIX_SYSTEM_PROMPT + "\n\n" + _lb.SYSTEM_PROMPT
        
        raw = self.bridge.ask(user_prompt, model=model)
        _lb.SYSTEM_PROMPT = _old_sp  # restore
        
        raw = re.sub(r"^```[\w]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        
        # Retry logic for JSON parse failures
        for attempt in range(3):
            try:
                data = json.loads(raw.strip())
                break
            except json.JSONDecodeError as e:
                if attempt < 2:
                    print(f"[matrix] JSON parse failed (attempt {attempt+1}): {e}")
                    print(f"[matrix] Retrying with completion request...")
                    retry_prompt = (
                        "Your previous response was truncated or had invalid JSON. "
                        "The error was: " + str(e) + "\n"
                        "Please regenerate the COMPLETE JSON response. "
                        "Output ONLY valid JSON, no markdown."
                    )
                    raw = self.bridge.ask(retry_prompt, model=model)
                    raw = re.sub(r"^```[\w]*\n?", "", raw, flags=re.MULTILINE)
                    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
                else:
                    print(f"[matrix] JSON parse failed after 3 attempts: {e}")
                    raise
        return MatrixResult(data, description, frame_end)

class MatrixResult:
    def __init__(self, data: dict, prompt: str, frame_end: int):
        self.data = data
        self.prompt = prompt
        self.frame_end = frame_end
        self.physics_type = data.get("physics_type", "unknown")
        self.base_code = data.get("base_scene_code", "")
        self.variants = data.get("parameter_matrix", [])

    def save(self, path: str):
        pathlib.Path(path).write_text(
            json.dumps({"prompt": self.prompt, "frame_end": self.frame_end, **self.data},
                       indent=2))
        print(f"[matrix] Saved {len(self.variants)} variants -> {path}")

    @classmethod
    def load(cls, path: str) -> "MatrixResult":
        data = json.loads(pathlib.Path(path).read_text())
        prompt = data.pop("prompt", "")
        frame_end = data.pop("frame_end", 72)
        return cls(data, prompt, frame_end)

    def prepare_variant_files(self, out_dir: pathlib.Path) -> list:
        """Write one .py file per variant, returns list of (variant_id, code_path, output_dir)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        result = []
        for v in self.variants:
            vid = v["variant_id"]
            vdir = out_dir / vid
            vdir.mkdir(parents=True, exist_ok=True)
            frame_path = str(vdir / "frame_####")
            # Inject paths into override code
            override = v.get("override_code", "")
            override = override.replace("__OUTPUT_PATH__", frame_path)
            full_code = (
                self.base_code
                    .replace("__FRAME_END__", str(self.frame_end))
                + "\n\n# === VARIANT: " + vid + " === " + v.get("quality_label","") + "\n"
                + override
            )
            code_path = out_dir / f"{vid}.py"
            code_path.write_text(full_code)
            result.append((vid, str(code_path), vdir, v.get("quality_label",""),
                           v.get("description","")))
        return result
