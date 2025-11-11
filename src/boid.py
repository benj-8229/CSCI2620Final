import math
import numpy as np
from random import random

FLOCK_DISTANCE = 30
AVOIDANCE_DISTANCE = 10


class Boid:
    def __init__(self, x_pos: int | float, y_pos: int | float, speed: float, sim: 'Simulation'):
        self.x_pos: float = float(x_pos)
        self.y_pos: float = float(y_pos)
        self.speed: float = speed
        self.sim: 'Simulation' = sim
        self.direction = np.array([0.0, 0.0], dtype=float)
        self.interpolated_dir = np.array([0.0, 0.0], dtype=float)

        self.idx = -1

    def move(self):
        self.interpolated_dir = self.lerp_vec(self.interpolated_dir, self.direction, 3 * self.sim.delta)
        self.interpolated_dir = self.normalize_vec(self.interpolated_dir)

        self.x_pos += self.interpolated_dir[0] * self.speed
        self.y_pos += self.interpolated_dir[1] * self.speed

        if self.sim.wrapping:
            self.x_pos = (self.x_pos + self.sim.x_size) % self.sim.x_size
            self.y_pos = (self.y_pos + self.sim.y_size) % self.sim.y_size

    def rotate(self, deg: float = 0):
        current_dir: float = Boid.vec2deg(self.direction)
        new_dir = Boid.deg2vec(current_dir + deg)
        self.direction = self.normalize_vec(new_dir)

    def steer(self, snapshot: list['Boid'], xs, ys, distances, dirs):
        # get a bool mask of all the distances in certain ranges
        mask_flock = (distances[self.idx] < FLOCK_DISTANCE)
        mask_avoid = (distances[self.idx] < AVOIDANCE_DISTANCE)
        # set self.idx to false to not include self
        mask_flock[self.idx] = False
        mask_avoid[self.idx] = False
        # build lists of boids based off masks (legacy)
        others = [snapshot[j] for j in np.nonzero(mask_flock)[0]]
        others_close = [snapshot[j] for j in np.nonzero(mask_avoid)[0]]

        aw = .3
        cw = .4
        sw = .8
        ww = .8
        wind = (self.sim.wind[int(self.y_pos), int(self.x_pos)])

        ax, ay = self.alignment(mask_flock, dirs, distances[self.idx])
        cx, cy = self.cohesion(others)
        sx, sy = self.separation(others_close)
        wx, wy = self.deg2vec(wind * 360)

        fx = ax * aw + cx * cw + sx * sw + wx * ww
        fy = ay * aw + cy * cw + sy * sw + wy * ww

        if fx*fx + fy*fy < 1e-5:
            return

        self.direction = self.normalize_vec(np.array([fx, fy], dtype=float))

    # steer towards the average direction of nearby boids
    def alignment(self, mask, dirs, distances) -> tuple[float, float]:
        directions = dirs[mask]
        if directions.size == 0:
            return (0.0, 0.0)

        t = distances[mask] / FLOCK_DISTANCE
        mag = (1.0 - t) ** 2
        scaled_dirs = directions * mag[:, None]

        return tuple(np.mean(scaled_dirs, axis=0))

    def cohesion(self, others: list['Boid']) -> tuple[float, float]:
        if not others:
            return (0.0, 0.0)

        if len(others) > 3:
            avg_vel = sum(b.speed for b in others) / len(others)
            self.speed = max(30 * self.sim.delta, self.lerp_float(self.speed, avg_vel, .75 * self.sim.delta))

        ax = self.wrapped_mean([b.x_pos for b in others], self.sim.x_size)
        ay = self.wrapped_mean([b.y_pos for b in others], self.sim.y_size)
        dx = self.sim.dx_between_boids(self, Boid(ax, ay, self.speed, self.sim))
        dy = self.sim.dy_between_boids(self, Boid(ax, ay, self.speed, self.sim))

        dist = math.hypot(dx, dy)

        if dist == 0.0:
            return (0.0, 0.0)

        n = np.array([dx / dist, dy / dist], dtype=float)
        magnitude = min(dist / FLOCK_DISTANCE, 1.0)
        n *= magnitude

        return (float(n[0]), float(n[1]))

    def separation(self, others: list['Boid']) -> tuple[float, float]:
        if len(others) == 0:
            return (0.0, 0.0)

        acc = np.array([0.0, 0.0], dtype=float)
        for other in others:
            dist = self.sim.dist_between_boids(self, other)
            t = dist / AVOIDANCE_DISTANCE
            if dist < AVOIDANCE_DISTANCE:
                magnitude = (1.0 - t) ** 2
                acc += -np.array([self.sim.dx_between_boids(self, other),
                                  self.sim.dy_between_boids(self, other)], dtype=float) * magnitude

        return (float(acc[0]), float(acc[1]))

    def wrapped_mean(self, values, size):
        angles = [(v / size) * 2 * math.pi for v in values]

        sx = sum(math.cos(a) for a in angles)
        sy = sum(math.sin(a) for a in angles)

        if sx == 0.0 and sy == 0.0:
            return size * 0.5  # fallback

        mean_angle = math.atan2(sy, sx)
        if mean_angle < 0.0:
            mean_angle += 2 * math.pi

        return (mean_angle / (2 * math.pi)) * size

    @staticmethod
    def deg2vec(deg: float):
        rad = np.radians(deg)
        return np.array([np.cos(rad), np.sin(rad)], dtype=float)

    @staticmethod
    def vec2deg(vec) -> float:
        v = np.asarray(vec, dtype=float)
        return float(np.degrees(np.arctan2(v[1], v[0])))

    @staticmethod
    def normalize_vec(vec):
        v = np.asarray(vec, dtype=float)
        m = float(np.linalg.norm(v))
        if m == 0.0:
            theta = random() * 2 * np.pi
            return np.array([np.cos(theta), np.sin(theta)])

        return v / m

    @staticmethod
    def lerp_float(a, b, t) -> float:
        # fix: correct linear interpolation
        return float(a + (b - a) * t)

    @staticmethod
    def lerp_vec(a, b, t):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return a + (b - a) * t

