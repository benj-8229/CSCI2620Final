import math
import numpy as np
from copy import deepcopy
from typing import Any
from PIL import Image
from boid import Boid
from hsv import hsv_to_rgb
from perlin import perlin
from concurrent.futures import ThreadPoolExecutor

easeInOut = lambda x: 16 * x * x * x * x * x if x < .5 else 1 - (-2 * x + 2)**5 / 2

BOID_COLOR = (255, 255, 255)
BOID_HEAD_COLOR = (245, 93, 227)

MOTION_BLUR = .85

class Simulation:
    def __init__(self, x_size: int = 350, y_size: int = 350, delta: float = 1 / 60, wrapping: bool = True, grid_size: int = 8):
        self.frame = 0

        self.x_size = x_size
        self.y_size = y_size
        self.delta = delta
        self.wrapping = wrapping
        self.grid_size = grid_size
        self.boids: list[Boid] = []

        self.light_kernel = self._make_light_kernel(radius=33, intensity=.7, beta=0.325)
        self.accum_buffer = np.zeros((self.y_size, self.x_size, 3), dtype=np.float32)

        self.noise_x = 0
        self.noise_y = 0
        self.noise = perlin(self.noise_x, self.noise_y, self.x_size, self.y_size)
        self.wind = perlin(self.noise_x, self.noise_y, self.x_size, self.y_size, scale=.01, octaves=6, persistence=.5, lacunarity=1.5, seed=100)

        # draw grid
        self.grid = Image.new("RGB", (self.x_size, self.y_size))
        raster: Any = self.grid.load()
        for x in range(0, self.x_size):
            for y in range(0, self.y_size, 8):
                raster[x, y] = (66, 66, 66)
                raster[y, x] = (66, 66, 66)
                # raster[x, y] = (33, 33, 32)
                # raster[y, x] = (33, 33, 32)

    def step(self):
        self.frame += 1

        self.noise_x += self.delta * 10
        self.noise_y += self.delta * 10

        if self.frame % 5 == 0:
            self.noise = perlin(self.noise_x, self.noise_y, self.x_size, self.y_size)

            self.wind = perlin(-self.noise_x * 5, -self.noise_y * 5, self.x_size, self.y_size, scale=.01, seed=100)
            self.wind = np.array([[easeInOut(j) for j in row] for row in self.wind])    
            # self.wind = self.noise * self.noise

        snapshot = []
        for boid in self.boids:
            boid.sim = None
            b = deepcopy(boid)
            b.sim = None
            boid.sim = self
            snapshot.append(b)

        xs = np.array([b.x_pos for b in self.boids], dtype=float)
        ys = np.array([b.y_pos for b in self.boids], dtype=float)
        positions = np.column_stack((xs, ys))  # shape (n, 2)
        distances = self.pairwise_distances(positions)

        dirs = np.array([b.interpolated_dir for b in self.boids])

        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     for boid in self.boids:
        #         executor.submit(lambda b=boid: (b.steer(snapshot, distances), b.move()))

        for boid in self.boids:
            boid.steer(snapshot, xs, ys, distances, dirs)
        for boid in self.boids:
            boid.move()

    def pairwise_distances(self, positions):
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape (n, n, 2)

        if self.wrapping:
            # Wrap differences into [-W/2, +W/2] and [-H/2, +H/2]
            delta[..., 0] = (delta[..., 0] + self.x_size/2) % self.x_size - self.x_size/2
            delta[..., 1] = (delta[..., 1] + self.y_size/2) % self.y_size - self.y_size/2

        dist_sq = np.einsum("ijk,ijk->ij", delta, delta)       # squared distances
        return np.sqrt(dist_sq)

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

    def boids_around_boid(self, boids: list[Boid], boid: Boid, r: float) -> list[Boid]:
        out = []

        for other in boids:
            d = self.dist_between_boids(boid, other)
            if other != boid and d > 1e-6 and d <= r:
                    out.append(other)

        return out

    def dist_between_boids(self, a: Boid, b: Boid) -> float:
        dx = b.x_pos - a.x_pos
        dy = b.y_pos - a.y_pos

        if not self.wrapping:
            return math.sqrt(dx**2 + dy**2)

        half_x = 0.5 * self.x_size
        half_y = 0.5 * self.y_size

        if dx >  half_x: dx -= self.x_size
        if dx < -half_x: dx += self.x_size
        if dy >  half_y: dy -= self.y_size
        if dy < -half_y: dy += self.y_size

        return math.sqrt(dx**2 + dy**2)

    def dx_between_boids(self, a: Boid, b: Boid) -> float:
        dx = b.x_pos - a.x_pos
        if not self.wrapping:
            return dx

        half = 0.5 * self.x_size
        if dx >  half: dx -= self.x_size
        if dx < -half: dx += self.x_size
        return dx

    def dy_between_boids(self, a: Boid, b: Boid) -> float:
        dy = b.y_pos - a.y_pos
        if not self.wrapping:
            return dy
        half = 0.5 * self.y_size
        if dy >  half: dy -= self.y_size
        if dy < -half: dy += self.y_size
        return dy

    def draw(self) -> Image.Image:
        grid = self.grid.copy()
        raster: Any = grid.load()

        toplayer: list = []
        lightmap = np.zeros((self.y_size, self.x_size, 3), dtype=np.float32)

        for boid in self.boids:
            h = (Boid.vec2deg(boid.interpolated_dir) + 360) % 360
            r, g, b = hsv_to_rgb(h, 1, 1)
            head_color = (int(r * 255), int(g * 255), int(b * 255))
    
            # save boid pos
            mapX, mapY = self.map_x(round(boid.x_pos)), self.map_y(round(boid.y_pos))
            toplayer.append((mapX, mapY, BOID_COLOR))

            # save boid heading
            headX = self.map_x(round(boid.x_pos + boid.direction[0]))
            headY = self.map_y(round(boid.y_pos + boid.direction[1]))
            toplayer.append((headX, headY, head_color))

            self.add_light(lightmap, round(boid.x_pos), round(boid.y_pos), tuple(c / 255 for c in head_color))

        # clamp lightmap between 0-1 and convert grid to an array in range 0-1
        noise = np.repeat(self.noise[..., None], 3, axis=-1)
        base = np.asarray(grid, dtype=np.float32) / 255.0
        base = np.clip(base + noise * .75, 0, 1)

        # add lightmap to motion blur buffer and then combine that with our base image
        self.accum_buffer = self.accum_buffer * MOTION_BLUR + lightmap * (1.0 - MOTION_BLUR)
        illumination = np.clip(self.accum_buffer * 1.2, 0, 1)
        combined = np.clip(base * (illumination + 0.2), 0, 2)

        # convert back to 0-255 image in RGB mode
        out = Image.fromarray((combined * 255).astype(np.uint8), mode="RGB")
        # out = Image.fromarray((np.repeat(self.wind[..., None], 3, axis=-1) * 255).astype(np.uint8), mode="RGB")
        raster = out.load()

        # redraw boid pixels so they're on top layer
        for x, y, color in toplayer:
            raster[x, y] = color

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

