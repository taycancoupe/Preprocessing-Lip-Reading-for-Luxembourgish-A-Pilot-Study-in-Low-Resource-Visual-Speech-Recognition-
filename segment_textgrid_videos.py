from pathlib import Path
import csv
import re
import subprocess
import argparse


ROOT_DEFAULT = r"D:\BSP\bsp_dataset_no_silence\Video\Christane"
SPEAKER = "Christane"


def read_textgrid(path: Path) -> str:
    for enc in ("utf-16", "utf-16-le", "utf-8-sig", "utf-8"):
        try:
            return path.read_text(encoding=enc).replace("\x00", "")
        except UnicodeError:
            continue
    return path.read_text(errors="ignore").replace("\x00", "")


def clean_transcript(text: str) -> str:
    text = text.replace('\\"', '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_sentences_tier(textgrid_path: Path):
    content = read_textgrid(textgrid_path)

    tier_match = re.search(
        r'name\s*=\s*"Sentences"(.*?)(?:item \[\d+\]:|\Z)',
        content,
        re.DOTALL,
    )

    if not tier_match:
        raise ValueError('Tier name = "Sentences" not found')

    tier = tier_match.group(1)

    interval_pattern = re.compile(
        r"intervals \[\d+\]:\s*"
        r"xmin\s*=\s*([0-9.]+)\s*"
        r"xmax\s*=\s*([0-9.]+)\s*"
        r'text\s*=\s*"(.*?)"',
        re.DOTALL,
    )

    intervals = []

    for match in interval_pattern.finditer(tier):
        start = float(match.group(1))
        end = float(match.group(2))
        transcript = clean_transcript(match.group(3))

        if not transcript:
            continue

        if end <= start:
            continue

        intervals.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "transcript": transcript,
        })

    return intervals


def find_single_file(folder: Path, pattern: str):
    files = list(folder.glob(pattern))
    return files[0] if files else None


def cut_video(ffmpeg, input_video, output_video, start, duration, overwrite=False):
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y" if overwrite else "-n",

        "-ss", f"{start:.3f}",
        "-i", str(input_video),
        "-t", f"{duration:.3f}",

        "-map", "0:v:0",
        "-map", "0:a:1",

        "-vf", "scale=1920:1080,fps=25,format=yuv420p,setsar=1",

        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "16",
        "-profile:v", "high",
        "-level", "4.1",
        "-g", "25",

        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "1",

        "-sn",
        "-dn",
        "-map_metadata", "-1",
        "-movflags", "+faststart",

        str(output_video),
    ]

    return subprocess.run(cmd, capture_output=True, text=True)

def process_session(session_dir: Path, ffmpeg: str, overwrite: bool):
    session_name = session_dir.name
    session_num = re.sub(r"\D+", "", session_name) or session_name

    mp4_path = find_single_file(session_dir, "*.mp4")
    textgrid_path = find_single_file(session_dir, "*.TextGrid") or find_single_file(session_dir, "*.textgrid")

    if not mp4_path:
        print(f"[SKIP] {session_name}: no MP4 found")
        return [], []

    if not textgrid_path:
        print(f"[SKIP] {session_name}: no TextGrid found")
        return [], []

    print(f"\n[SESSION] {session_name}")
    print(f"Video: {mp4_path.name}")
    print(f"TextGrid: {textgrid_path.name}")

    try:
        intervals = parse_sentences_tier(textgrid_path)
    except Exception as e:
        print(f"[ERROR] Could not parse TextGrid: {e}")
        return [], [{
            "session": session_name,
            "clip_id": "",
            "error": str(e),
            "output_path": "",
        }]

    output_dir = session_dir / "no_silence"
    output_dir.mkdir(exist_ok=True)

    rows = []
    failures = []

    for idx, item in enumerate(intervals, start=1):
        clip_id = f"{SPEAKER}_session{session_num}_{idx:04d}"
        output_path = output_dir / f"{clip_id}.mp4"

        if output_path.exists() and not overwrite:
            print(f"[EXISTS] {output_path.name}")
        else:
            result = cut_video(
                ffmpeg=ffmpeg,
                input_video=mp4_path,
                output_video=output_path,
                start=item["start"],
                duration=item["duration"],
                overwrite=overwrite,
            )

            if result.returncode != 0:
                print(f"[FAIL] {clip_id}: {result.stderr.strip()}")
                failures.append({
                    "session": session_name,
                    "clip_id": clip_id,
                    "error": result.stderr.strip(),
                    "output_path": str(output_path),
                })
                continue

            print(f"[OK] {clip_id}")

        rows.append({
            "speaker": SPEAKER,
            "session": session_name,
            "clip_id": clip_id,
            "start_time": f"{item['start']:.3f}",
            "end_time": f"{item['end']:.3f}",
            "duration": f"{item['duration']:.3f}",
            "transcript": item["transcript"],
            "output_path": str(output_path),
        })

    return rows, failures


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT_DEFAULT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)

    all_rows = []
    all_failures = []

    session_dirs = sorted(
    [p for p in root.iterdir() if p.is_dir() and "session" in p.name.lower()]
    )

    print(f"Root: {root}")
    print(f"Found sessions: {len(session_dirs)}")

    for session_dir in session_dirs:
        rows, failures = process_session(session_dir, args.ffmpeg, args.overwrite)
        all_rows.extend(rows)
        all_failures.extend(failures)

        if rows:
            session_csv = session_dir / "no_silence" / "metadata.csv"
            write_csv(
                session_csv,
                rows,
                ["speaker", "session", "clip_id", "start_time", "end_time", "duration", "transcript", "output_path"],
            )

    global_csv = root / "metadata_all_sessions.csv"
    write_csv(
        global_csv,
        all_rows,
        ["speaker", "session", "clip_id", "start_time", "end_time", "duration", "transcript", "output_path"],
    )

    if all_failures:
        failure_csv = root / "failed_clips.csv"
        write_csv(
            failure_csv,
            all_failures,
            ["session", "clip_id", "error", "output_path"],
        )

    print("\nDone.")
    print(f"Metadata: {global_csv}")
    print(f"Failed clips: {len(all_failures)}")


if __name__ == "__main__":
    main()