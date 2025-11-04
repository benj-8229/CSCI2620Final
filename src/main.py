import subprocess, time, sys
from pathlib import Path
from random import randrange, random
from simulation import Simulation
from boid import Boid

FPS = 45
SECONDS = 10
FRAMES = FPS * SECONDS
BOIDS = 50 * 2
BOID_SPEED = 40
SCALE = 8
GRID_SIZE = 10
SIZE = 150
OUTPUT_PATH = "output/output.mp4"

print(f"Calculating {FPS * SECONDS} frames with a delta of {1.0/FPS}")

frames_dir = Path(OUTPUT_PATH).with_suffix("").parent / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)

sim = Simulation(SIZE, SIZE, (1.0 / FPS))
sim.boids = [Boid(randrange(0, SIZE - 1), randrange(0, SIZE - 1), 1, sim) for _ in range(BOIDS)]
for boid in sim.boids:
    boid.direction = Boid.deg2vec(randrange(0, 360))
    boid.interpolated_dir = boid.direction
    boid.speed = (BOID_SPEED + (BOID_SPEED * 2/3) * random()) * (1.0 / FPS)
    #boid.speed = BOID_SPEED * (1.0 / FPS)


start_time = time.time()
for i in range(FPS * SECONDS):
    sim.step()
    # frame = sim.draw(SCALE)
    frame = sim.draw(1)
    frame.convert("RGB").save(frames_dir / f"frame_{i:06d}.png")

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
