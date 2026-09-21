use crate::tpes::Tpes;
use crate::uss::Z9;

/// The ID 2 Couplet is the universal baseline unit cell (1 USS Unit).
/// All spatial bounds, energy levels, and scales are measured as integer multiples of this reference unit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(align(64))]
pub struct Couplet {
    pub id: u64,
    pub particle_dna: Tpes,
    pub energy: Z9,
    pub pressure: Z9,
    pub flux: [Z9; 27],
}

impl Couplet {
    pub const BASELINE_ID: u64 = 2;
    pub const PLANCK_PER_USS: u64 = 1024; // Discrete ruler scale factor: 1 USS Unit = 1024 Planck units

    pub fn new(id: u64, particle_dna: Tpes) -> Self {
        Self {
            id,
            particle_dna,
            energy: Z9::new(particle_dna.net_charge().abs()),
            pressure: Z9::ZERO,
            flux: [Z9::ZERO; 27],
        }
    }

    /// Creates a baseline ID 2 Couplet.
    pub fn new_baseline(particle_dna: Tpes) -> Self {
        Self::new(Self::BASELINE_ID, particle_dna)
    }

    /// Ruler math: Converts discrete spatial scale distance in USS units to Planck scale units.
    pub fn uss_to_planck(uss_units: u64) -> u64 {
        uss_units * Self::PLANCK_PER_USS
    }

    /// Ruler math: Converts Planck units to integer USS units.
    pub fn planck_to_uss(planck_units: u64) -> u64 {
        planck_units / Self::PLANCK_PER_USS
    }

    /// Calculates Manhattan distance between two integer 3D coordinates in USS units.
    pub fn uss_distance(pos1: (i64, i64, i64), pos2: (i64, i64, i64)) -> u64 {
        ((pos1.0 - pos2.0).abs() + (pos1.1 - pos2.1).abs() + (pos1.2 - pos2.2).abs()) as u64
    }

    /// Computes the aggregate Z9 flux across all 27 directional headings.
    pub fn total_flux(&self) -> Z9 {
        self.flux.iter().fold(Z9::ZERO, |acc, &f| acc + f)
    }

    /// Updates discrete flux state based on Z/9Z ring mechanics across all headings.
    pub fn step_flux(&mut self) {
        self.flux = [Z9::update_flux(self.energy, self.pressure); 27];
        // Discrete pressure cycle shift
        self.pressure += Z9::ONE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_couplet_creation() {
        let dna = Tpes::new(2, 2, 1, 1);
        let couplet = Couplet::new_baseline(dna);
        assert_eq!(couplet.id, 2);
        assert_eq!(couplet.particle_dna, dna);
        assert_eq!(couplet.energy, Z9::new(1));
    }

    #[test]
    fn test_reference_ruler_math() {
        assert_eq!(Couplet::uss_to_planck(1), 1024);
        assert_eq!(Couplet::uss_to_planck(5), 5120);
        assert_eq!(Couplet::planck_to_uss(2048), 2);
        assert_eq!(Couplet::planck_to_uss(1000), 0);

        let dist = Couplet::uss_distance((0, 0, 0), (2, 3, 4));
        assert_eq!(dist, 9);
    }

    #[test]
    fn test_couplet_flux_step() {
        let dna = Tpes::new(2, 2, 1, 1);
        let mut couplet = Couplet::new_baseline(dna);
        assert_eq!(couplet.energy, Z9::new(1));
        assert_eq!(couplet.pressure, Z9::ZERO);

        couplet.step_flux();
        // flux = 1 + 0*2 = 1 across all 27 headings, pressure becomes 1
        assert_eq!(couplet.flux, [Z9::new(1); 27]);
        assert_eq!(couplet.pressure, Z9::new(1));

        couplet.step_flux();
        // flux = 1 + 1*2 = 3 across all 27 headings, pressure becomes 2
        assert_eq!(couplet.flux, [Z9::new(3); 27]);
        assert_eq!(couplet.pressure, Z9::new(2));
    }

    #[test]
    fn test_couplet_alignment() {
        assert_eq!(std::mem::align_of::<Couplet>(), 64);
    }
}
