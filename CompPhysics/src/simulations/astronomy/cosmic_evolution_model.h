#pragma once
#include <algorithm>
#include <array>
#include <cmath>

// Curated, piecewise-linear educational tracks. Ages are years since the
// beginning of each selected scenario, never a shared cosmic clock.
namespace evolution {
struct Track {
    const char* name;
    const char* subtitle;
    const char* scale;
    const char* stages[5];
    const char* notes[5];
    std::array<double,6> years;
    double mass;
    double temperature;
};
inline const std::array<std::array<Track,3>,4> tracks{{
    {{
        {"Rocky world", "Silicate rock / iron core", "Earth-scale world",
         {"Dust nursery", "Planetesimals", "Giant impacts", "Core & crust", "Ocean world"},
         {"Grains orbit a newborn star. Gentle collisions let some particles stick and grow.",
          "Pebbles and planetesimals accumulate. Gravity increasingly controls collisions.",
          "Embryos collide and release heat. The growing world can develop a magma ocean.",
          "Dense metal sinks inward. A mantle and crust develop as the surface cools.",
          "One possible outcome: a rocky planet with water. Oceans require suitable conditions."},
         {0,1e6,5e6,5e7,1e8,5e8},1,288},
        {"Gas giant", "Rock / ice core + hydrogen", "Jupiter-scale world",
         {"Icy grains", "Core assembly", "Gas capture", "Envelope growth", "Banded giant"},
         {"Beyond the snow line, frozen volatiles add solid material to the disk.",
          "An illustrative core-accretion pathway builds a massive solid core.",
          "The core binds a gaseous envelope while the surrounding disk still contains gas.",
          "Rapid envelope growth produces a hydrogen-rich giant; the disk then disperses.",
          "The giant cools and contracts. Rotation organizes atmospheric flow into bands."},
         {0,5e5,1e6,3e6,5e6,1e8},318,130},
        {"Ice giant", "Volatiles / rock / light gases", "Neptune-scale world",
         {"Outer disk", "Frozen building blocks", "Core growth", "Modest envelope", "Cold blue world"},
         {"A cold outer disk contains rock and volatile-rich solids.",
          "Icy building blocks accumulate in a slowly evolving outer system.",
          "A volatile-rich interior grows, with substantial rock mixed into the planet.",
          "A modest hydrogen-helium envelope forms; runaway gas capture does not occur here.",
          "Methane absorbs red light. The blue color is illustrative; interiors remain uncertain."},
         {0,1e6,3e6,6e6,1e7,5e8},17,72}
    }},
    {{
        {"Red dwarf", "0.2 solar masses / M type", "Stellar radius exaggerated",
         {"Molecular cloud", "Protostar", "Hydrogen fusion", "Long quiet life", "Future remnant"},
         {"A dense region of a cold molecular cloud becomes gravitationally unstable.",
          "Contraction heats the center; a disk feeds the protostar before sustained fusion.",
          "A cool red dwarf settles onto the main sequence and consumes hydrogen slowly.",
          "Low mass permits a lifetime of trillions of years, far beyond the present cosmic age.",
          "A helium white dwarf is a theoretical distant outcome. No such isolated track is observed yet."},
         {0,1e6,5e8,1e12,5e12,6e12},0.2,3200},
        {"Sun-like star", "1 solar mass / G type", "Stellar radius exaggerated",
         {"Molecular cloud", "Protostar", "Main sequence", "Red giant", "White dwarf"},
         {"Gravity gathers gas and dust into a dense core inside a molecular cloud.",
          "Accretion and contraction provide energy before sustained core hydrogen fusion.",
          "Hydrogen fuses into helium. Pressure and gravity maintain approximate balance.",
          "Core hydrogen runs low; the envelope expands. Later phases include helium burning.",
          "The envelope is shed into a planetary nebula. The hot white dwarf slowly cools."},
         {0,1e6,5e7,1e10,1.2e10,1.3e10},1,5772},
        {"Massive star", "25 solar masses / O type", "Stellar radius exaggerated",
         {"Dense cloud", "Rapid accretion", "Blue main sequence", "Supergiant", "Core collapse"},
         {"A massive cloud core supplies material to a rapidly growing protostar.",
          "Strong radiation and outflows compete with accretion during stellar assembly.",
          "A hot blue star burns through its fuel far faster than a low-mass star.",
          "Successive burning stages build heavier elements. The envelope can swell dramatically.",
          "Core collapse may leave a neutron star or black hole. This track illustrates a black-hole outcome."},
         {0,1e5,1e6,6e6,7e6,7.1e6},25,35000}
    }},
    {{
        {"Stellar remnant", "Massive-star collapse pathway", "Horizon visually enlarged",
         {"Massive progenitor", "Fuel exhausted", "Core collapse", "Fallback disk", "Black hole"},
         {"A massive star approaches the end of its nuclear-burning life.",
          "An iron-rich core cannot gain energy through further fusion. Pressure support fails.",
          "Collapse occurs in seconds. An explosion is possible, but not every collapse makes one.",
          "Some material falls back. Gas with angular momentum can form a hot accretion disk.",
          "A 10-solar-mass remnant is chosen for display. Real outcomes depend on mass loss and composition."},
         {0,6e6,7e6,7e6+1.0/31557600,7e6+1,7e6+1000},10,0},
        {"Growing seed", "Illustrative intermediate-mass growth", "Horizon visually enlarged",
         {"Seed remnant", "Gas supply", "Accretion", "Feedback", "Intermediate mass"},
         {"A seed black hole is assumed to exist at the start of this scenario.",
          "Surrounding gas supplies the seed. Continuous feeding is an idealization.",
          "Radiatively efficient growth is illustrated with an exponential mass law.",
          "Radiation and outflows can heat gas and interrupt feeding in real systems.",
          "Intermediate-mass black holes bridge stellar and supermassive scales; formation routes vary."},
         {0,2e7,5e7,1e8,1.5e8,2e8},100,0},
        {"Galactic nucleus", "Illustrative supermassive growth", "Horizon visually enlarged",
         {"Nuclear seed", "Gas inflow", "Active nucleus", "Bright quasar", "Quiet nucleus"},
         {"A pre-existing million-solar-mass seed is assumed, not created by a single ordinary star.",
          "Galactic gas can lose angular momentum and move toward the nucleus.",
          "A hot accretion flow radiates as matter approaches the black hole.",
          "Sustained feeding can power a quasar. Jets require additional magnetic conditions.",
          "When fuel becomes scarce, the nucleus dims. The black hole remains even without a bright disk."},
         {0,3e7,8e7,1.5e8,2e8,3e8},1e6,0}
    }},
    {{
        {"Spiral galaxy", "Rotating disk / young & old stars", "Tens of thousands of light-years",
         {"Gas in a halo", "First star clusters", "Rotating disk", "Spiral structure", "Mature spiral"},
         {"Gas cools within a dark-matter halo; the halo itself is not shown as luminous matter.",
          "Dense gas forms clusters. Young massive stars light up their surroundings.",
          "Angular momentum supports a growing disk while older stars populate a central bulge.",
          "Spiral structure organizes gas and young stars. Stars orbit through the pattern.",
          "A mature spiral keeps forming stars where cold gas remains. This is one schematic history."},
         {0,2e8,1e9,3e9,7e9,1.2e10},1e11,0},
        {"Elliptical galaxy", "Spheroid / older stellar population", "Tens of thousands of light-years",
         {"Gas-rich ancestors", "Starburst", "Assembly", "Gas depleted", "Stellar spheroid"},
         {"Gas-rich progenitors provide the starting material for this illustrative assembly history.",
          "An early burst forms many stars. The most massive stars evolve quickly.",
          "Mergers and dynamical mixing can redistribute stellar orbits into a spheroid.",
          "The supply of cold gas declines. Star formation slows, and the population reddens.",
          "Older stars dominate a smooth elliptical. Ellipticals have diverse assembly histories."},
         {0,2e8,1e9,3e9,7e9,1.2e10},3e11,0},
        {"Irregular galaxy", "Clumpy gas / ongoing star formation", "Thousands of light-years",
         {"Gas reservoir", "Patchy starbirth", "Uneven assembly", "Bright knots", "Irregular system"},
         {"A small, asymmetric gas reservoir begins forming stars within its halo.",
          "Star formation lights isolated clumps rather than a single ordered disk.",
          "Feedback and interactions disturb the gas and the stellar distribution.",
          "Young clusters illuminate scattered regions. A stable spiral need not appear.",
          "The result is a gas-rich irregular system, one of many possible morphologies."},
         {0,1e8,5e8,2e9,5e9,1e10},1e9,0}
    }}
}};
inline int stage(double p) { return std::min(4,int(std::clamp(p,0.0,1.0)*5)); }
inline double age(const Track& t,double p) {
    p=std::clamp(p,0.0,1.0)*5;
    int i=std::min(4,int(p));
    return t.years[i]+(t.years[i+1]-t.years[i])*(p-i);
}
inline double blackHoleMass(int variant,double p) {
    const double seed=tracks[2][variant].mass;
    return variant==0 ? seed : seed*std::exp(age(tracks[2][variant],p)/4.5e7*0.55);
}
inline double horizonKm(double solarMass) { return 2.95325*solarMass; }
} // namespace evolution
