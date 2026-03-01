"""
Threat & Interceptor Catalog: Realistic weapon system parameters.

Based on publicly available specifications for Iron Dome and
common regional threats. All values approximate — this is a
learning simulation, not classified data.

THREATS (things we're trying to stop):
  - Qassam:       Unguided rocket, 40kg, short range
  - Grad:         Unguided MLRS rocket, 66kg, medium range
  - Mortar:       High-angle indirect fire, 13kg
  - Cruise:       Guided cruise missile, 1500kg, subsonic
  - UAV:          Small drone, 50kg, slow

INTERCEPTORS (things we use to stop them):
  - Tamir:        Iron Dome interceptor, 90kg, Mach 2.2
  - Stunner:      David's Sling interceptor, 400kg, Mach 7.5
  - Arrow3:       Arrow 3 exoatmospheric interceptor, 2000kg, Mach 9
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .ballistics import PropulsionConfig


@dataclass
class ThreatProfile:
    """Template for a threat type."""
    name: str
    dry_mass: float          # kg (without fuel)
    total_mass: float        # kg (fully loaded)
    cross_section: float     # m² frontal area
    drag_coefficient: float  # Cd
    max_accel: float         # m/s² (G-limit * 9.81)
    guided: bool             # Can it steer after launch?
    propulsion: PropulsionConfig
    description: str = ""


@dataclass
class InterceptorProfile:
    """Template for an interceptor type."""
    name: str
    tier: str                # "iron_dome", "davids_sling", "arrow"
    dry_mass: float          # kg
    total_mass: float        # kg
    cross_section: float     # m²
    drag_coefficient: float  # Cd
    max_accel: float         # m/s²
    propulsion: PropulsionConfig
    min_range: float         # meters (minimum engagement range)
    max_range: float         # meters (maximum engagement range)
    max_speed: float         # m/s (terminal velocity)
    description: str = ""


# ─── THREAT CATALOG ──────────────────────────────────────────────

QASSAM = ThreatProfile(
    name="Qassam",
    dry_mass=25.0,
    total_mass=40.0,
    cross_section=0.015,
    drag_coefficient=0.5,
    max_accel=0.0,  # Unguided — no steering
    guided=False,
    propulsion=PropulsionConfig(
        thrust=3000.0,       # ~3kN
        burn_time=1.5,       # Short burn
        specific_impulse=200,
        fuel_mass=15.0,
    ),
    description="Short-range unguided rocket (5-17km range)",
)

GRAD = ThreatProfile(
    name="Grad BM-21",
    dry_mass=40.0,
    total_mass=66.0,
    cross_section=0.02,
    drag_coefficient=0.45,
    max_accel=0.0,  # Unguided
    guided=False,
    propulsion=PropulsionConfig(
        thrust=8000.0,       # ~8kN
        burn_time=2.0,
        specific_impulse=220,
        fuel_mass=26.0,
    ),
    description="MLRS unguided rocket (20-40km range)",
)

MORTAR = ThreatProfile(
    name="120mm Mortar",
    dry_mass=10.0,
    total_mass=13.0,
    cross_section=0.012,
    drag_coefficient=0.6,  # Higher drag (not streamlined)
    max_accel=0.0,  # Unguided
    guided=False,
    propulsion=PropulsionConfig(
        thrust=12000.0,      # Very high initial thrust (propellant charge)
        burn_time=0.05,      # Near-instantaneous (barrel launch)
        specific_impulse=150,
        fuel_mass=3.0,
    ),
    description="High-angle indirect fire round (4-8km range)",
)

CRUISE_MISSILE = ThreatProfile(
    name="Cruise Missile",
    dry_mass=1000.0,
    total_mass=1500.0,
    cross_section=0.3,
    drag_coefficient=0.35,
    max_accel=39.2,  # ~4G (can maneuver)
    guided=True,
    propulsion=PropulsionConfig(
        thrust=5000.0,       # Sustained turbojet-like
        burn_time=120.0,     # Long-endurance
        specific_impulse=800,
        fuel_mass=500.0,
    ),
    description="Subsonic guided cruise missile (300+km range)",
)

UAV = ThreatProfile(
    name="Small UAV",
    dry_mass=35.0,
    total_mass=50.0,
    cross_section=0.1,
    drag_coefficient=0.5,
    max_accel=19.6,  # ~2G
    guided=True,
    propulsion=PropulsionConfig(
        thrust=500.0,        # Electric motor equivalent
        burn_time=1800.0,    # 30 minutes
        specific_impulse=0,  # Not applicable (electric)
        fuel_mass=15.0,      # Battery weight
    ),
    description="Low-slow reconnaissance/attack drone",
)

# ─── INTERCEPTOR CATALOG ─────────────────────────────────────────

TAMIR = InterceptorProfile(
    name="Tamir",
    tier="iron_dome",
    dry_mass=60.0,
    total_mass=90.0,
    cross_section=0.03,
    drag_coefficient=0.25,
    max_accel=294.3,  # ~30G (very agile)
    propulsion=PropulsionConfig(
        thrust=15000.0,      # ~15kN boost
        burn_time=4.0,       # 4 second boost
        specific_impulse=260,
        fuel_mass=22.0,
        sustain_thrust=4000.0,   # ~4kN sustainer
        sustain_burn_time=6.0,   # 6 second sustain
        sustain_fuel_mass=8.0,
    ),
    min_range=4000.0,        # 4 km
    max_range=70000.0,       # 70 km
    max_speed=748.0,         # Mach 2.2
    description="Iron Dome interceptor",
)

STUNNER = InterceptorProfile(
    name="Stunner",
    tier="davids_sling",
    dry_mass=250.0,
    total_mass=400.0,
    cross_section=0.06,
    drag_coefficient=0.2,
    max_accel=196.2,  # ~20G
    propulsion=PropulsionConfig(
        thrust=80000.0,      # ~80kN booster
        burn_time=5.0,
        specific_impulse=280,
        fuel_mass=100.0,
        sustain_thrust=20000.0,  # ~20kN sustainer
        sustain_burn_time=8.0,
        sustain_fuel_mass=50.0,
    ),
    min_range=40000.0,       # 40 km
    max_range=300000.0,      # 300 km
    max_speed=2551.5,        # Mach 7.5
    description="David's Sling hit-to-kill interceptor",
)

ARROW3 = InterceptorProfile(
    name="Arrow 3",
    tier="arrow",
    dry_mass=1200.0,
    total_mass=2000.0,
    cross_section=0.15,
    drag_coefficient=0.15,
    max_accel=147.1,  # ~15G
    propulsion=PropulsionConfig(
        thrust=300000.0,     # ~300kN (three-stage booster)
        burn_time=12.0,
        specific_impulse=300,
        fuel_mass=800.0,
    ),
    min_range=100000.0,      # 100 km
    max_range=2400000.0,     # 2400 km
    max_speed=3061.8,        # Mach 9
    description="Arrow 3 exoatmospheric interceptor",
)


# ─── CATALOGS ─────────────────────────────────────────────────────

THREAT_CATALOG: dict[str, ThreatProfile] = {
    "qassam": QASSAM,
    "grad": GRAD,
    "mortar": MORTAR,
    "cruise_missile": CRUISE_MISSILE,
    "uav": UAV,
}

INTERCEPTOR_CATALOG: dict[str, InterceptorProfile] = {
    "tamir": TAMIR,
    "stunner": STUNNER,
    "arrow3": ARROW3,
}
