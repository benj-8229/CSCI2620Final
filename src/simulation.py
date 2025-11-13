import numpy as np
import numpy.typing as npt
import random
from typing import Any
from PIL import Image
from hsv import hsv_to_rgb
from perlin import perlin

easeInOut = lambda x: 16 * x * x * x * x * x if x < .5 else 1 - (-2 * x + 2)**5 / 2

BOID_COLOR = (255, 255, 255)
BOID_HEAD_COLOR = (245, 93, 227)
MOTION_BLUR = .85

FLOCK_DISTANCE = 30
AVOIDANCE_DISTANCE = 10
EPSILON = np.finfo(np.float32).eps

class Simulation:
    def __init__(self, x_size: int = 300, y_size: int = 300, delta: float = 1 / 60, wrapping: bool = True, grid_size: int = 8):
        self.frame = 0

        self.x_size = x_size
        self.y_size = y_size
        self.delta = delta
        self.wrapping = wrapping
        self.grid_size = grid_size

        self.light_kernel = self._make_light_kernel(radius=33, intensity=.7, beta=0.325)
        self.accum_buffer = np.zeros((self.y_size, self.x_size, 3), dtype=np.float32)

        self.noise_x = 0
        self.noise_y = 0
        self.noise = perlin(self.noise_x, self.noise_y, self.x_size, self.y_size)
        self.wind = perlin(self.noise_x, self.noise_y, self.x_size, self.y_size, scale=.01, octaves=6, persistence=.5, lacunarity=1.5, seed=100)

        self.boid_n: int = 0
        self.boid_pos: npt.NDArray = np.zeros((1,))
        self.boid_speed: npt.NDArray = np.zeros((1,))
        self.boid_dir: npt.NDArray = np.zeros((1,))
        self.boid_lerpdir: npt.NDArray = np.zeros((1,))

        # draw grid
        self.grid = Image.new("RGB", (self.x_size, self.y_size))
        raster: Any = self.grid.load()
        for x in range(0, self.x_size):
            for y in range(0, self.y_size, 8):
                raster[x, y] = (66, 66, 66)
                raster[y, x] = (66, 66, 66)

    def initialize_boids(self, n, max_speed):
        self.boid_n = n
        self.boid_pos = np.zeros((n, 2)).astype(np.float32)
        self.boid_speed = np.zeros((n,)).astype(np.float32)
        self.boid_dir = np.zeros((n,2)).astype(np.float32)

        for i in range(n):
            self.boid_pos[i] = np.array([random.randrange(0, self.x_size), random.randrange(0, self.y_size)], dtype=np.float32)
            self.boid_speed[i] = random.randrange(int(max_speed), int(max_speed * 1.5))
            # self.boid_speed[i] = max_speed
            rad = random.random() * 2 * np.pi
            self.boid_dir[i] = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)

    def step(self):
        self.frame += 1

        self.noise_x += self.delta * 10
        self.noise_y += self.delta * 10

        # if self.frame % 5 == 0:
        #     self.noise = perlin(self.noise_x, self.noise_y, self.x_size, self.y_size)
        #     self.wind = perlin(-self.noise_x * 5, -self.noise_y * 5, self.x_size, self.y_size, scale=.01, seed=100)
        #     self.wind = np.array([[easeInOut(j) for j in row] for row in self.wind])    
        #     # self.wind = self.noise * self.noise

        aw = .7
        cw = .4
        sw = .3
        # ww = .8

        distances = self.pairwise_distances(self.boid_pos)
        new_dir = np.empty_like(self.boid_dir)
        new_speed = np.empty_like(self.boid_speed)

        flock_masks: Any = (distances < FLOCK_DISTANCE)
        avoid_masks: Any = (distances < AVOIDANCE_DISTANCE)
        np.fill_diagonal(flock_masks, False)
        np.fill_diagonal(avoid_masks, False)

        delta = self.boid_pos[:, None, :] - self.boid_pos[None, :, :]
        avoid_mask_exp = avoid_masks[..., None]
        delta_avoid = delta * avoid_mask_exp
        weights = (1 - np.sqrt(distances) / AVOIDANCE_DISTANCE) ** 2
        weights = weights * avoid_masks
        weights = weights[..., None]

        sv = (delta_avoid * weights).sum(axis=1) / (weights.sum(axis=1) + EPSILON)

        for i in range(self.boid_n):
            flock = np.copy(flock_masks[i])
            # avoid = np.copy(avoid_masks[i])

            flock_pos = self.boid_pos[flock]
            flock_speed = self.boid_speed[flock]
            flock_dir = self.boid_dir[flock]
            flock_distances = distances[i][flock]

            # avoid_pos = self.boid_pos[avoid]
            # avoid_distances = distances[i][avoid]


            # alignment
            av = np.copy(self.boid_dir[i])
            if flock_dir.size > 0:
                t = flock_distances / FLOCK_DISTANCE
                mag = (1.0 - t) ** 2
                align_dirs = flock_dir * mag[:, None]
                av += np.mean(align_dirs, axis=0)
                norm = np.linalg.norm(av)
                if norm > EPSILON:
                    av /= norm

            # cohesion position
            cv = np.zeros(2, dtype=np.float32)
            if flock_pos.size > 0:
                cap = self.wrapped_mean_position(flock_pos)
                delta = cap - self.boid_pos[i]

                dist = np.linalg.norm(delta)
                if dist > EPSILON:
                    mag = min(dist / FLOCK_DISTANCE, 1.0)
                    cv = (delta / dist) * mag

            # cohesion speed
            if flock_speed.size > 0:
                csa = np.mean(flock_speed)
                new_speed[i] = self.boid_speed[i] + (csa - self.boid_speed[i]) * (.1 * self.delta)
                new_speed[i] = min(self.boid_speed[i], 40)

            # separation
            # sv = np.zeros(2, dtype=np.float32)
            # if avoid_pos.size > 0:
            #     delta = self.boid_pos[i] - avoid_pos
            #     weights = (1.0 - (avoid_distances / AVOIDANCE_DISTANCE)) ** 2
            #     sv = np.mean(delta * weights[:, None], axis=0)

            #     norm = np.linalg.norm(sv)
            #     if norm > EPSILON:
            #         sv /= norm

            new_dir[i] = av * aw + cv * cw + sv[i] * sw

        # compute wind flow
        # wind_angles = self.wind * 2 * np.pi  # same shape (H, W)
        # wind_vec = np.stack((np.cos(wind_angles), np.sin(wind_angles)), axis=-1)  # shape (H, W, 2)
        # ix = np.clip(self.boid_pos[:, 0].astype(int), 0, self.x_size - 1)
        # iy = np.clip(self.boid_pos[:, 1].astype(int), 0, self.y_size - 1)
        # local_wind = wind_vec[iy, ix]  # shape (boid_n, 2)
        # self.boid_dir += local_wind * ww

        turn_rate = 10 * self.delta
        dir_next = self.boid_dir + (new_dir - self.boid_dir) * turn_rate
        norms = np.linalg.norm(dir_next, axis=1, keepdims=True)
        dir_next = np.divide(dir_next, norms, out=np.zeros_like(dir_next), where=norms > EPSILON)

        pos_next = self.boid_pos + dir_next * new_speed[:, None] * self.delta
        pos_next[:, 0] = np.mod(pos_next[:, 0], self.x_size)
        pos_next[:, 1] = np.mod(pos_next[:, 1], self.y_size)

        self.boid_pos = pos_next
        self.boid_dir = dir_next

    def pairwise_distances(self, positions: npt.NDArray):
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape (n, n, 2)

        if self.wrapping:
            # Wrap differences into [-W/2, +W/2] and [-H/2, +H/2]
            delta[..., 0] = (delta[..., 0] + self.x_size/2) % self.x_size - self.x_size/2
            delta[..., 1] = (delta[..., 1] + self.y_size/2) % self.y_size - self.y_size/2

        dist_sq = np.einsum("ijk,ijk->ij", delta, delta)       # squared distances
        return np.sqrt(dist_sq)

    def wrapped_mean_position(self, positions: npt.NDArray):
        if positions.size == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        sizes = np.array([self.x_size, self.y_size], dtype=np.float32)
        angles = (positions / sizes) * (2 * np.pi)  # (N, 2) angles in radians

        # Compute circular mean for each axis independently
        sx = np.sum(np.cos(angles), axis=0)
        sy = np.sum(np.sin(angles), axis=0)

        # Fallback for degenerate cases
        mask = (sx == 0) & (sy == 0)
        mean_angle = np.arctan2(sy, sx)
        mean_angle[mask] = np.pi  # equivalent to mid-point

        # Normalize angle to [0, 2π)
        mean_angle = np.mod(mean_angle, 2 * np.pi)

        # Convert back to linear coordinates
        mean_pos = (mean_angle / (2 * np.pi)) * sizes

        return mean_pos.astype(np.float32)

    def map_x(self, x):
        if self.wrapping:
            x = (x + self.x_size) % self.x_size
        else:
            x = max(0, min(x, self.x_size - 1))
        return x

    def map_y(self, y):
        if self.wrapping:
            y = (y + self.y_size) % self.y_size
        else:
            y = max(0, min(y, self.y_size - 1))
        return y

    def draw(self) -> Image.Image:
        base = np.asarray(self.grid, dtype=np.float32)
        # boids = np.zeros((self.y_size, self.x_size, 3), dtype=np.float32)
        # lightmap = np.zeros((self.y_size, self.x_size, 3), dtype=np.float32)

        for i in range(self.boid_n):
            # Grab position and direction from arrays
            x_pos, y_pos = self.boid_pos[i]
            dir_x, dir_y = self.boid_dir[i]

            # Compute heading hue from direction vector
            angle_deg = (np.degrees(np.arctan2(dir_y, dir_x)) + 360) % 360
            r, g, b = hsv_to_rgb(angle_deg, 1.0, 1.0)
            head_color = (int(r*255), int(g*255), int(b*255))

            # Map body pixel
            mapX = self.map_x(round(x_pos))
            mapY = self.map_y(round(y_pos))
            base[mapX, mapY] = BOID_COLOR

            # Map heading pixel
            headX = self.map_x(round(x_pos + dir_x))
            headY = self.map_y(round(y_pos + dir_y))
            base[headX, headY] = head_color

            # Add light contribution
            # self.add_light(
            #     lightmap,
            #     round(x_pos),
            #     round(y_pos),
            #     (r, g, b)
            # )

        # self.accum_buffer = self.accum_buffer * MOTION_BLUR + lightmap * (1.0 - MOTION_BLUR)
        # illumination = np.clip(self.accum_buffer * 1.4, 0, 255)
        # combined = np.clip((base + illumination * .5) / 255, 0, 255)
        # combined = base + illumination * .5)

        # convert back to 0-255 image in RGB mode
        out = Image.fromarray((base).astype(np.uint8), mode="RGB")

        return out

    def add_light(self, lightmap: np.ndarray, x: int, y: int, color: tuple[float, float, float]):
        # k = kernel
        # r = radius
        # h, w = size of our lightmap grid
        k = self.light_kernel
        r = k.shape[0] // 2
        h, w, _ = lightmap.shape

        col = np.array(color, np.float32)

        # generate grid of coordinates that wraps around the lightmap grid
        ys = (np.arange(-r, r + 1) + y) % h
        xs = (np.arange(-r, r + 1) + x) % w
        # convert coordinate grids to 2d grid
        Y, X = np.meshgrid(ys, xs, indexing='ij')

        # vector indexing to paint entire image in one step, still trying to fully understand
        # k[..., None] adds an empty dimension which allows multiplying by col
        # Y, X, are both vectors. so indexing by vectors does something?
        lightmap[Y, X] += k[..., None] * col


    def _make_light_kernel(self, radius, intensity, beta):
        # create 2d grid from -radius to radius+1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]

        # d =  2d distance grid (circular, 0 is at center, radius at edge)
        d = np.sqrt(x*x + y*y)

        # t = normalized distance grid from 0-1 (0 at center, 1 at edge)
        t = np.clip(d / radius, 0, 1)

        # inverse with smoothstep and falloff, so 1 is at center and 0 is at edge
        smooth = 1.0 - (3.0*t*t - 2.0*t*t*t)
        inv = 1.0 / (1.0 + (beta * d)**2)
        falloff = intensity * smooth * inv

        # clip edges that are greater than radius due to generous indexing
        falloff[d > radius] = 0.0

        return falloff.astype(np.float32)

