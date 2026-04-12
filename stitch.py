"""
stitch.py — ffmpeg scene stitcher

Usage:
    python stitch.py                         # stitch all scene dirs in output_dir
    python stitch.py scene_001 scene_002     # specific scenes in order
    python stitch.py --output my_film.mp4
    python stitch.py --fps 30 --crf 20
"""
import os, glob, pathlib, subprocess, argparse, yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("scenes", nargs="*",
                   help="Scene folder names to include (default: all)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--fps",    default=None, type=int)
    p.add_argument("--crf",    default=None, type=int)
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def frames_to_clip(frame_dir, clip_path, fps, crf):
    frames = sorted(glob.glob(str(frame_dir / "frame_*.png")))
    if not frames:
        frames = sorted(glob.glob(str(frame_dir / "*.png")))
    if not frames:
        print(f"[stitch] No frames in {frame_dir}, skipping")
        return False

    sample = pathlib.Path(frames[0]).stem
    digits = len(sample.replace("frame_", ""))
    pattern = str(frame_dir / f"frame_%0{digits}d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-crf",  str(crf),
        "-pix_fmt", "yuv420p",
        clip_path
    ]
    print(f"[stitch] Encoding {frame_dir.name} ({len(frames)} frames)")
    subprocess.run(cmd, check=True, capture_output=True)
    return True


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    bcfg = cfg["blender"]
    pcfg = cfg["pipeline"]

    out_dir = pathlib.Path(bcfg["output_dir"]).expanduser()
    fps = args.fps or bcfg.get("fps", 24)
    crf = args.crf or pcfg.get("ffmpeg_crf", 18)
    final = args.output or pcfg.get("output_video", "final.mp4")

    if args.scenes:
        scene_dirs = [out_dir / s for s in args.scenes if (out_dir / s).is_dir()]
    else:
        scene_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir()])

    if not scene_dirs:
        print("[stitch] No scene directories found.")
        return

    clips = []
    for sd in scene_dirs:
        clip = str(sd) + "_clip.mp4"
        if frames_to_clip(sd, clip, fps, crf):
            clips.append(clip)

    if not clips:
        print("[stitch] No clips to stitch.")
        return

    list_file = str(out_dir / "concat_list.txt")
    with open(list_file, "w") as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        final
    ]
    print(f"[stitch] Concatenating {len(clips)} clips -> {final}")
    subprocess.run(cmd, check=True)
    print(f"[stitch] Output: {final}")


if __name__ == "__main__":
    main()
