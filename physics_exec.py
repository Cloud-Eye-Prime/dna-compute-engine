"""
physics_exec.py — Runs INSIDE blender --background.
Executes LLM-generated physics bpy code (base scene + variant overrides).

Called as: blender --background --python physics_exec.py -- --code /path/to/variant.py
"""
import sys, traceback, pathlib


def get_arg(flag):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


code_path = get_arg("--code")
if not code_path or not pathlib.Path(code_path).exists():
    print(f"[physics_exec] ERROR: --code not found: {code_path}")
    sys.exit(1)

code = pathlib.Path(code_path).read_text()
print(f"[physics_exec] Running variant: {pathlib.Path(code_path).stem}  ({len(code)} chars)")

try:
    import bpy
    # Set render engine from environment or default EEVEE
    import os
    engine = os.environ.get("BLENDER_ENGINE", "BLENDER_EEVEE_NEXT")
    bpy.context.scene.render.engine = engine

    ns = {"bpy": bpy, "__file__": code_path}
    exec(compile(code, code_path, "exec"), ns)
    print(f"[physics_exec] SUCCESS: {pathlib.Path(code_path).stem}")
except Exception as e:
    print(f"[physics_exec] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
