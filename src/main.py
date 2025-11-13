import subprocess, time, sys
from pathlib import Path
from simulation import Simulation
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

FPS = 60
SECONDS = 10
FRAMES = FPS * SECONDS
BOIDS = 50
BOID_SPEED = 40
SCALE = 8
GRID_SIZE = 13
SIZE = 125
OUTPUT_PATH = "output/output.mp4"


print(f"Calculating {FPS * SECONDS} frames with a delta of {1.0/FPS}")


frames_dir = Path(OUTPUT_PATH).with_suffix("").parent / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)


# Initialize sim and boids
sim = Simulation(SIZE, SIZE, (1.0 / FPS), wrapping=True)
sim.initialize_boids(BOIDS, BOID_SPEED)


# Set up thread pool and callback function for frame renderers
executor = ThreadPoolExecutor(max_workers=32)
completed = 0
start_time = time.time()
lock = Lock()
def draw_callback(_):
    global completed
    with lock:
        completed += 1


for i in range(FRAMES):
    sim.step()
    frame = sim.draw().convert("RGB")

    job = executor.submit(
        frame.save, frames_dir / f"frame_{i:06d}.png"
    )
    job.add_done_callback(draw_callback)

    completed += 1

    # progress info
    elapsed = time.time() - start_time
    fps_est = (i + 1) / elapsed if elapsed > 0 else 0
    bar_len = 30
    filled_len = int(bar_len * (i + 1) / FRAMES)
    bar = "█" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write(
            f"\r[{bar}] Frame {i+1}/{FRAMES} {fps_est:5.1f} FPS {elapsed:5.1f}"
    )
    sys.stdout.flush()

executor.shutdown()
print("\n")

print(f"Rendering {FPS * SECONDS} frames...")
cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", str(frames_dir / "frame_%06d.png"),
    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    "-vf", f"scale={SIZE * SCALE}:{SIZE * SCALE}",
    "-sws_flags", "neighbor",
    OUTPUT_PATH
]
subprocess.run(cmd, check=True)
