"""
Enrichment Module - Uranium Enrichment Cascade Models

This module provides models for uranium enrichment calculations including:
- Value function calculations
- Separative Work Unit (SWU) calculations
- Cascade material balance
- Feed, product, and tails calculations
"""

import math
from dataclasses import dataclass
from typing import Optional


U235_ZAID = 922350
U238_ZAID = 922380

NATURAL_U235_ENRICHMENT = 0.00711
NATURAL_U238_ENRICHMENT = 0.99289


def value_function(x: float) -> float:
    """
    Calculate the enrichment value function V(x).

    The value function is a thermodynamic quantity that expresses the
    separative work required to achieve a given enrichment.

    V(x) = (2x - 1) * ln(x / (1 - x))

    Parameters:
        x: Mass fraction of U-235 (0 < x < 1)

    Returns:
        The value function V(x)

    Raises:
        ValueError: If x is not in valid range (0, 1)
    """
    if x <= 0 or x >= 1:
        raise ValueError(f"Enrichment x must be between 0 and 1, got {x}")

    if x == 0.5:
        return 0.0

    return (2 * x - 1) * math.log(x / (1 - x))


def calculate_swu(
    feed: Optional[float] = None,
    product: Optional[float] = None,
    tails: Optional[float] = None,
    x_feed: float = NATURAL_U235_ENRICHMENT,
    x_product: float = 0.05,
    x_tails: float = 0.0025,
) -> float:
    """
    Calculate Separative Work Units (SWU) required for enrichment.

    At least two of (feed, product, tails) must be provided.
    The third will be calculated from material balance.

    SWU = P * V(x_p) + T * V(x_t) - F * V(x_f)

    Parameters:
        feed: Mass of feed material in kg U
        product: Mass of product material in kg U
        tails: Mass of tails (depleted uranium) in kg U
        x_feed: Feed enrichment (mass fraction of U-235)
        x_product: Product enrichment (mass fraction of U-235)
        x_tails: Tails enrichment (mass fraction of U-235)

    Returns:
        Separative Work Units required

    Raises:
        ValueError: If insufficient parameters provided
    """
    v_feed = value_function(x_feed)
    v_product = value_function(x_product)
    v_tails = value_function(x_tails)

    known_count = sum(x is not None for x in [feed, product, tails])

    if known_count < 2:
        raise ValueError("At least two of feed, product, or tails must be provided")

    if product is not None and feed is not None:
        tails = feed - product
    elif product is not None and tails is not None:
        feed = product + tails
    elif feed is not None and tails is not None:
        product = feed - tails
    else:
        raise ValueError("Could not determine missing parameter")

    if feed <= 0 or product <= 0 or tails < 0:
        raise ValueError(f"Invalid mass values: F={feed}, P={product}, T={tails}")

    swu = product * v_product + tails * v_tails - feed * v_feed
    return max(0, swu)


def calculate_feed_ratio(x_feed: float, x_product: float, x_tails: float) -> float:
    """
    Calculate the feed to product ratio (F/P).

    From material balance: F * x_f = P * x_p + T * x_t
    And: F = P + T

    F/P = (x_p - x_t) / (x_f - x_t)

    Parameters:
        x_feed: Feed enrichment
        x_product: Product enrichment
        x_tails: Tails enrichment

    Returns:
        Feed to product ratio
    """
    return (x_product - x_tails) / (x_feed - x_tails)


def calculate_tails_ratio(x_feed: float, x_product: float, x_tails: float) -> float:
    """
    Calculate the tails to product ratio (T/P).

    T/P = (x_f - x_p) / (x_t - x_p)

    Parameters:
        x_feed: Feed enrichment
        x_product: Product enrichment
        x_tails: Tails enrichment

    Returns:
        Tails to product ratio
    """
    return (x_feed - x_product) / (x_tails - x_product)


@dataclass
class EnrichmentCascade:
    """
    Model for an enrichment cascade.

    Attributes:
        feed_enrichment: U-235 mass fraction in feed (default: 0.711% natural)
        product_enrichment: U-235 mass fraction in product
        tails_enrichment: U-235 mass fraction in tails (depleted uranium)
        feed_mass: Feed mass in kg U
        separation_factor: Stage separation factor (alpha)
    """

    feed_enrichment: float = NATURAL_U235_ENRICHMENT
    product_enrichment: float = 0.05
    tails_enrichment: float = 0.0025
    feed_mass: float = 1000.0
    separation_factor: float = 1.2

    def __post_init__(self):
        if not 0 < self.feed_enrichment < 1:
            raise ValueError("Feed enrichment must be between 0 and 1")
        if not 0 < self.product_enrichment < 1:
            raise ValueError("Product enrichment must be between 0 and 1")
        if not 0 < self.tails_enrichment < 1:
            raise ValueError("Tails enrichment must be between 0 and 1")
        if self.product_enrichment <= self.tails_enrichment:
            raise ValueError("Product enrichment must be greater than tails")
        if self.feed_enrichment <= self.tails_enrichment:
            raise ValueError("Feed enrichment must be greater than tails")

    @property
    def feed_ratio(self) -> float:
        """Calculate feed to product ratio."""
        return calculate_feed_ratio(
            self.feed_enrichment, self.product_enrichment, self.tails_enrichment
        )

    @property
    def tails_ratio(self) -> float:
        """Calculate tails to product ratio."""
        return calculate_tails_ratio(
            self.feed_enrichment, self.product_enrichment, self.tails_enrichment
        )

    @property
    def product_mass(self) -> float:
        """Calculate product mass in kg U."""
        return self.feed_mass / self.feed_ratio

    @property
    def tails_mass(self) -> float:
        """Calculate tails mass in kg U."""
        return self.feed_mass - self.product_mass

    @property
    def swu_total(self) -> float:
        """Calculate total SWU required."""
        return calculate_swu(
            feed=self.feed_mass,
            product=self.product_mass,
            tails=self.tails_mass,
            x_feed=self.feed_enrichment,
            x_product=self.product_enrichment,
            x_tails=self.tails_enrichment,
        )

    @property
    def swu_per_feed(self) -> float:
        """Calculate SWU per kg of feed."""
        return self.swu_total / self.feed_mass

    @property
    def swu_per_product(self) -> float:
        """Calculate SWU per kg of product."""
        return self.swu_total / self.product_mass

    def summary(self) -> dict:
        """Return summary of cascade parameters."""
        return {
            "feed_mass_kg": self.feed_mass,
            "product_mass_kg": self.product_mass,
            "tails_mass_kg": self.tails_mass,
            "feed_enrichment_pct": self.feed_enrichment * 100,
            "product_enrichment_pct": self.product_enrichment * 100,
            "tails_enrichment_pct": self.tails_enrichment * 100,
            "total_swu": self.swu_total,
            "swu_per_kg_feed": self.swu_per_feed,
            "swu_per_kg_product": self.swu_per_product,
        }


def typical_pwr_fuel_requirements(
    reactor_power_mwe: float,
    burnup_gwd: float,
    enrichment_pct: float = 4.5,
    capacity_factor: float = 0.9,
) -> dict:
    """
    Calculate typical fuel requirements for a PWR reactor.

    Parameters:
        reactor_power_mwe: Reactor power in MWe
        burnup_gwd: Discharge burnup in GWd/tU
        enrichment_pct: Target fuel enrichment in weight percent
        capacity_factor: Plant capacity factor

    Returns:
        Dictionary with fuel requirements
    """
    burnup_mwd_per_kg = burnup_gwd * 1000

    annual_energy_mwh = reactor_power_mwe * 8760 * capacity_factor

    efficiency = 0.33
    annual_thermal_energy = annual_energy_mwh / efficiency

    kg_u_per_gwd = 1e6 / burnup_mwd_per_kg
    annual_heavy_metal = annual_thermal_energy / (burnup_mwd_per_kg * 1000)

    cascade = EnrichmentCascade(
        feed_mass=annual_heavy_metal,
        product_enrichment=enrichment_pct / 100,
        x_tails=0.0025,
    )

    return {
        "reactor_power_mwe": reactor_power_mwe,
        "annual_fresh_fuel_kg": cascade.product_mass,
        "annual_swu_required": cascade.swu_total,
        "depleted_uranium_kg": cascade.tails_mass,
        "enrichment_pct": enrichment_pct,
        "burnup_gwd_t": burnup_gwd,
    }


if __name__ == "__main__":
    cascade = EnrichmentCascade(
        feed_mass=1000, product_enrichment=0.045, tails_enrichment=0.0025
    )

    print("=== Enrichment Cascade Results ===")
    for key, value in cascade.summary().items():
        print(f"{key}: {value:.4f}")

    print("\n=== PWR Fuel Requirements (1000 MWe) ===")
    pwr_req = typical_pwr_fuel_requirements(1000, 50)
    for key, value in pwr_req.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
