"""Planar magnetic tracks and energy/momentum/charge-balanced toy collisions.

Positions: m. Momenta: GeV/c. Masses: GeV/c². B: tesla, positive out of plane.
The common clock is slowed so c is drawn as 3 m per animation second. Tracks
are stable particles in vacuum; energy loss, radiation and decays are omitted.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

MAGNETIC_K = .299792458
DISPLAY_C = 3.0
DETECTOR_RADIUS = 4.0
LAYERS = (.65, 1.2, 1.9, 2.7, 3.6)
SPECIES = (("Electrons / positrons", .00051099895, 1),
           ("Muons / antimuons", .1056583755, 1),
           ("Protons / antiprotons", .938272088, 1),
           ("Photons", 0., 0))
POSITIVE = (100, 186, 250)
NEGATIVE = (245, 186, 105)
NEUTRAL = (220, 225, 235)


def rotate(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    x, y = vector
    return np.array([c*x-s*y, s*x+c*y])


@dataclass
class Track:
    pos: np.ndarray
    momentum: np.ndarray
    mass: float
    charge: int
    name: str = "particle"
    age: float = 0.0
    alive: bool = True
    fade: float = 2.2
    points: deque = field(default_factory=lambda: deque(maxlen=650))
    layers_hit: set = field(default_factory=set)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float).copy()
        self.momentum = np.asarray(self.momentum, dtype=float).copy()
        self.points.append(self.pos.copy())

    @property
    def energy(self):
        return math.sqrt(float(self.momentum@self.momentum)+self.mass*self.mass)

    @property
    def color(self):
        return POSITIVE if self.charge > 0 else NEGATIVE if self.charge < 0 else NEUTRAL

    def radius(self, magnetic_field):
        denominator = MAGNETIC_K*abs(self.charge*magnetic_field)
        return math.inf if denominator < 1e-12 else float(np.linalg.norm(self.momentum))/denominator

    def advance(self, dt, magnetic_field):
        """Exact uniform-field arc, preserving momentum magnitude and energy."""
        energy = self.energy
        if energy <= 0:
            return
        velocity = DISPLAY_C*self.momentum/energy
        omega = -self.charge*MAGNETIC_K*magnetic_field*DISPLAY_C/energy
        angle = omega*dt
        if abs(omega) < 1e-10:
            self.pos += velocity*dt
        else:
            sine, cosine = math.sin(angle), math.cos(angle)
            vx, vy = velocity
            self.pos += np.array([vx*sine+vy*(cosine-1), vx*(1-cosine)+vy*sine])/omega
            self.momentum = rotate(self.momentum, angle)
        self.points.append(self.pos.copy())


def circle_crossings(start, end, radius):
    delta = end-start
    a = float(delta@delta)
    if a < 1e-16:
        return []
    b = 2*float(start@delta)
    c = float(start@start)-radius*radius
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return []
    root = math.sqrt(max(0, discriminant))
    values = [(-b-root)/(2*a), (-b+root)/(2*a)]
    return [start+t*delta for t in values if 1e-8 < t <= 1]


@dataclass
class Collision:
    energy: float
    vertex: np.ndarray
    momenta: list
    mass: float = .13957039

    @property
    def outgoing_energy(self):
        return sum(math.sqrt(float(p@p)+self.mass**2) for p in self.momenta)

    @property
    def momentum_residual(self):
        return np.sum(self.momenta, axis=0)


def toy_collision(energy, rng, vertex=(0., 0.), pairs=3, axis=0.):
    """Neutral CM beams -> charge-balanced pions. Kinematics, not probabilities.

    Center random 2D momenta to make their sum zero, then find a common momentum
    scale that puts the massive final state on the requested energy shell.
    No cross section or shower generator is implied. Charges alternate +/−.
    """
    mass = .13957039
    if pairs < 1 or not np.isfinite(energy) or energy <= 2*pairs*mass:
        raise ValueError("Collision energy must exceed the selected final-state rest energy")
    vectors = rng.normal(size=(2*pairs, 2))
    vectors -= vectors.mean(axis=0)
    vectors[-1] = -vectors[:-1].sum(axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    if norms.sum() < 1e-12:
        raise ValueError("Degenerate final-state momentum sample")
    low, high = 0., energy/float(norms.sum())
    for _ in range(75):
        factor = (low+high)/2
        total = float(np.sqrt(mass*mass+(factor*norms)**2).sum())
        if total > energy:
            high = factor
        else:
            low = factor
    momenta = [rotate(p*((low+high)/2), axis) for p in vectors]
    return Collision(energy, np.asarray(vertex, dtype=float).copy(), momenta)


class Studio:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.tracks = []
        self.hits = []
        self.magnetic_field = 1.4
        self.momentum = .45
        self.collision_energy = 6.0
        self.species = 0
        self.pending = None
        self.last_event = None
        self.event_flash = 0.
        self.events = 0

    def clear(self):
        self.tracks.clear()
        self.hits.clear()
        self.pending = None
        self.last_event = None
        self.event_flash = 0.

    def release(self, position=(0., 0.), angle=math.pi/2):
        name, mass, charge = SPECIES[self.species]
        for sign in (1, -1):
            theta = angle if charge else angle+sign*.13
            momentum = self.momentum*np.array([math.cos(theta), math.sin(theta)])
            self.tracks.append(Track(position, momentum, mass, charge*sign, name))
        self.tracks = self.tracks[-64:]

    def collide(self, vertex=(0., 0.), axis=0.):
        if self.pending is not None:
            return False
        self.clear()
        vertex = np.asarray(vertex, dtype=float).copy()
        length = np.linalg.norm(vertex)
        if length > 1.5:
            vertex *= 1.5/length
        self.pending = dict(vertex=vertex, axis=float(axis), progress=0., energy=self.collision_energy)
        return True

    def step(self, dt):
        dt = float(np.clip(dt, 0, .08))
        self.event_flash = max(0, self.event_flash-dt)
        self.hits = [(p, color, life-dt) for p, color, life in self.hits if life > dt]
        if self.pending is not None:
            pending = self.pending
            pending['progress'] += dt/.5
            if pending['progress'] >= 1:
                event = toy_collision(pending['energy'], self.rng, pending['vertex'], axis=pending['axis'])
                for i, momentum in enumerate(event.momenta):
                    self.tracks.append(Track(event.vertex, momentum, event.mass, 1 if i%2 == 0 else -1, "pion"))
                self.last_event = event
                self.events += 1
                self.event_flash = .5
                self.pending = None
        for track in self.tracks:
            track.age += dt
            if track.alive:
                remaining = dt
                while remaining > 1e-8:
                    step = min(.012, remaining)
                    old = track.pos.copy()
                    track.advance(step, self.magnetic_field)
                    for i, radius in enumerate(LAYERS):
                        if i not in track.layers_hit:
                            crossings = circle_crossings(old, track.pos, radius)
                            if crossings:
                                self.hits.append((crossings[0], track.color, 1.2))
                                track.layers_hit.add(i)
                    remaining -= step
                    if np.linalg.norm(track.pos) > DETECTOR_RADIUS or track.age > 6.5:
                        track.alive = False
                        break
            else:
                track.fade -= dt
        self.tracks = [t for t in self.tracks if t.alive or t.fade > 0]
