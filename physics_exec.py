"""
physics_exec.py -- Runs INSIDE blender --background.
Executes LLM-generated physics bpy code (base scene + variant overrides).

Called as: blender --background --python physics_exec.py -- --code /path/to/variant.py
"""
import sys, traceback, pathlib, os


def get_arg(flag):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def setup_gpu():
    """Enable CUDA GPU rendering for Cycles. Silent fallback to CPU."""
    try:
        import bpy
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            if device.type == 'CUDA':
                device.use = True
            elif device.type == 'CPU':
                device.use = False
    except Exception:
        pass  # CPU fallback is fine


code_path = get_arg("--code")
if not code_path or not pathlib.Path(code_path).exists():
    print(f"[physics_exec] ERROR: --code not found: {code_path}")
    sys.exit(1)

code = pathlib.Path(code_path).read_text()
print(f"[physics_exec] Running variant: {pathlib.Path(code_path).stem}  ({len(code)} chars)")

try:
    import bpy

    # Force Cycles renderer (EEVEE fails headless on Linux)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = int(os.environ.get("BLENDER_SAMPLES", "64"))
    setup_gpu()

    ns = {"bpy": bpy, "__file__": code_path}
    exec(compile(code, code_path, "exec"), ns)
    print(f"[physics_exec] SUCCESS: {pathlib.Path(code_path).stem}")
except Exception as e:
    print(f"[physics_exec] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)