"""
blender_scene_exec.py
Invoked as: blender --background --python blender_scene_exec.py -- --code /path/to/scene.py

Receives LLM-generated bpy code via a temp file and executes it inside Blender.
"""
import sys, os, traceback, pathlib


def get_arg(flag):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


code_path = get_arg("--code")
if not code_path or not pathlib.Path(code_path).exists():
    print(f"[blender_exec] ERROR: --code file not found: {code_path}")
    sys.exit(1)

code = pathlib.Path(code_path).read_text()
print(f"[blender_exec] Executing {len(code)} chars of bpy code from {code_path}")

try:
    import bpy
    exec(compile(code, code_path, "exec"), {"bpy": bpy, "__file__": code_path})
    print("[blender_exec] SUCCESS")
except Exception as e:
    print(f"[blender_exec] EXEC FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
