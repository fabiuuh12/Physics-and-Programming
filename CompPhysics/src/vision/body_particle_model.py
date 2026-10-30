"""Illustrative 2D transport in a moving, homogeneous water-like silhouette.

Coordinates are pixels, with an explicit model scale of 0.05 cm/pixel. The
silhouette is a target shape, never an inferred anatomical thickness. Outside
it is vacuum. Animation velocity and colors are for display only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
import math

import numpy as np

CM_PER_PIXEL = 0.05
KINDS = ("alpha", "beta", "gamma")
# MeV, not an isotope spectrum. Each selected beam is monoenergetic.
ENERGIES = {"alpha": (2.0, 10.0, 5.0), "beta": (0.1, 3.0, 1.0), "gamma": (0.05, 1.0, 0.662)}
COLORS = {"alpha": (105, 192, 250), "beta": (245, 185, 115), "gamma": (225, 190, 225)}
# Total mass attenuation coefficients for liquid water, NIST XrayMassCoef.
# Density is 1 g/cm^3, so the numerical linear coefficient is cm^-1.
# https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
PHOTON_ENERGIES = np.array([.001, .005, .01, .02, .03, .05, .08, .1, .2, .3, .5, .8, 1.])
PHOTON_MU = np.array([4078., 42.58, 5.329, .8096, .3756, .2269, .1837, .1707, .1370, .1186, .09687, .07865, .07072])


def charged_range(kind, energy):
    """Approximate path range in cm of unit-density water, not a dose model.

    Alpha: illustrative E^1.5 law anchored at 35 micrometers for 5 MeV.
    Beta: empirical electron mass-range approximation; straightened path budget.
    These are deliberately not advertised as ASTAR/ESTAR tabulations.
    """
    e = np.maximum(np.asarray(energy, dtype=float), 1e-6)
    if kind == "alpha":
        return 0.0035*(e/5)**1.5
    if kind == "beta":
        return np.where(e <= 2.5, .412*e**(1.265-.0954*np.log(e)), .530*e-.106)
    raise ValueError("Charged range requires alpha or beta")


@lru_cache(maxsize=128)
def range_curve(kind, energy):
    energies = np.geomspace(.001, energy, 70)
    return charged_range(kind, energies), energies


def photon_mu(energy):
    return float(np.exp(np.interp(np.log(max(energy, .001)), np.log(PHOTON_ENERGIES), np.log(PHOTON_MU))))


def compton_mu(energy):
    """Free-electron Klein–Nishina total cross section times water electron density."""
    k = max(energy, .001)/.51099895
    log = math.log1p(2*k)
    term = ((1+k)/k**2)*(2*(1+k)/(1+2*k)-log/k) + log/(2*k) - (1+3*k)/(1+2*k)**2
    cross_section = 2*math.pi*(2.8179403262e-13)**2*term
    electrons_per_cm3 = 10*6.02214076e23/18.01528
    return cross_section*electrons_per_cm3


def compton_event(energy, rng):
    """Sample polar angle with KN weighting; depict it in a 2D plane.

    Recoil electron energy leaves the tracked photon. Secondary electrons and
    out-of-plane paths are not tracked. The animation is not a 3D transport code.
    """
    for _ in range(10000):
        cosine = rng.uniform(-1, 1)
        ratio = 1/(1+(energy/.51099895)*(1-cosine))
        weight = ratio**2*(ratio+1/ratio-(1-cosine*cosine))
        if rng.random()*2 <= weight:
            sign = -1 if rng.random() < .5 else 1
            return energy*ratio, sign*math.acos(cosine)
    raise RuntimeError("Compton rejection sampler failed to converge")


@dataclass
class Particle:
    pos: np.ndarray
    direction: float
    energy: float
    initial_energy: float
    kind: str
    range_left: float = 0.0
    optical_left: float = 0.0
    trail: deque = field(default_factory=lambda: deque(maxlen=70))
    touched: bool = False
    scattered: bool = False
    alive: bool = True
    age: float = 0.0
    fade: float = 0.65


class Transport:
    def __init__(self, kind="alpha", energy=None, seed=None, cm_per_pixel=CM_PER_PIXEL):
        if kind not in KINDS:
            raise ValueError("Unknown particle type")
        if not np.isfinite(cm_per_pixel) or cm_per_pixel <= 0:
            raise ValueError("Material scale must be finite and positive")
        self.kind = kind
        self.energy = ENERGIES[kind][2] if energy is None else float(energy)
        if not np.isfinite(self.energy) or self.energy <= 0:
            raise ValueError("Energy must be finite and positive")
        self.cm_per_pixel = cm_per_pixel
        self.rng = np.random.default_rng(seed)
        self.particles = []
        self.flashes = []
        self.counts = dict(through=0, scattered=0, stopped=0, missed=0, expired=0)
        self.accumulator = 0.0

    def reset(self, kind=None, energy=None):
        if kind is not None:
            if kind not in KINDS:
                raise ValueError("Unknown particle type")
            self.kind = kind
        self.energy = ENERGIES[self.kind][2] if energy is None else float(energy)
        if not np.isfinite(self.energy) or self.energy <= 0:
            raise ValueError("Energy must be finite and positive")
        self.particles.clear()
        self.flashes.clear()
        self.counts = dict.fromkeys(self.counts, 0)
        self.accumulator = 0.0

    def emit(self, position, direction=0.0):
        if len(self.particles) >= 100:
            return None
        particle = Particle(np.asarray(position, dtype=float).copy(), direction,
                            self.energy, self.energy, self.kind,
                            range_left=float(charged_range(self.kind, self.energy)) if self.kind != "gamma" else 0,
                            optical_left=float(self.rng.exponential()))
        particle.trail.append(particle.pos.copy())
        self.particles.append(particle)
        return particle

    def finish(self, particle, outcome):
        if not particle.alive:
            return
        particle.alive = False
        self.counts[outcome] += 1

    def advance(self, particle, distance, mask):
        """Subpixel path steps resolve silhouette edges and charged stopping ranges."""
        h, w = mask.shape
        budget = distance
        while budget > 1e-7 and particle.alive:
            step = min(.5, budget)
            unit = np.array([math.cos(particle.direction), math.sin(particle.direction)])
            midpoint = particle.pos+unit*(step*.5)
            x, y = midpoint
            material = 0 <= x < w and 0 <= y < h and mask[int(y), int(x)]
            move = step
            event = None
            if material:
                particle.touched = True
                length_cm = step*self.cm_per_pixel
                if particle.kind != "gamma":
                    if particle.range_left <= length_cm:
                        move = max(0, particle.range_left/self.cm_per_pixel)
                        particle.range_left = 0
                        particle.energy = 0
                        event = "stop"
                    else:
                        particle.range_left -= length_cm
                        # Approximate energy from the same monotone range law.
                        ranges, energies = range_curve(particle.kind, particle.initial_energy)
                        particle.energy = float(np.interp(particle.range_left, ranges, energies))
                        if particle.kind == "beta":
                            # Qualitative multiple scattering, not a tissue-specific fit.
                            angle = self.rng.normal(0, .5*math.sqrt(length_cm/max(particle.range_left, .01)))
                            particle.direction += angle
                            particle.scattered = True
                else:
                    mu = photon_mu(particle.energy)
                    optical = mu*length_cm
                    if particle.optical_left <= optical:
                        move = particle.optical_left/(mu*self.cm_per_pixel)
                        # Residual channels (including coherent removal) are
                        # grouped as removal from this simplified tracked beam.
                        if self.rng.random() < min(1, compton_mu(particle.energy)/mu):
                            particle.energy, angle = compton_event(particle.energy, self.rng)
                            particle.direction += angle
                            particle.scattered = True
                            particle.optical_left = float(self.rng.exponential())
                            event = "scatter"
                        else:
                            particle.energy = 0
                            event = "stop"
                    else:
                        particle.optical_left -= optical
            particle.pos += unit*move
            budget -= max(move, 1e-7)
            if event:
                self.flashes.append((particle.pos.copy(), .45, event))
                if event == "stop":
                    self.finish(particle, "stopped")
            x, y = particle.pos
            if not (-5 <= x <= w+5 and -5 <= y <= h+5):
                outcome = "missed" if not particle.touched else "scattered" if particle.scattered else "through"
                self.finish(particle, outcome)
        particle.trail.append(particle.pos.copy())

    def step(self, mask, dt, source=None, emit=True):
        dt = float(np.clip(dt, 0, .08))
        self.flashes = [(p, life-dt, event) for p, life, event in self.flashes if life > dt]
        if emit and source is not None:
            self.accumulator += dt*11
            while self.accumulator >= 1:
                self.accumulator -= 1
                jitter = np.array([0., self.rng.normal(0, 14)])
                self.emit(np.asarray(source)+jitter, self.rng.normal(0, .015))
        for particle in self.particles:
            particle.age += dt
            if particle.alive:
                self.advance(particle, 300*dt, mask)
                if particle.age > 8:
                    self.finish(particle, "expired")
            else:
                particle.fade -= dt
        self.particles = [p for p in self.particles if p.alive or p.fade > 0]
