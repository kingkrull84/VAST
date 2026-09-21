use crate::lattice::couplet::Couplet;
use crate::lattice::octree::Octree;
use crate::tpes::Tpes;

/// Stamps a single Electron (T:1, P:0, E:1, S:1) at the given origin inside the Octree.
/// Injects an ID 1 Vacuum/Sink couplet at origin acting as a topological sink.
pub fn stamp_electron(octree: &mut Octree, origin: (i64, i64, i64)) {
    let electron_dna = Tpes::new(1, 0, 1, 1);
    let couplet = Couplet::new(1, electron_dna); // ID 1 Vacuum/Sink
    octree.insert(origin, couplet);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::octree::AABB;

    #[test]
    fn test_stamp_electron() {
        let bounds = AABB::new(-10, 10, -10, 10, -10, 10);
        let mut octree = Octree::new(bounds, 4, 4);

        let origin = (0, 0, 0);
        stamp_electron(&mut octree, origin);

        let t = octree.telemetry();
        assert_eq!(t.total_couplets, 1);

        let c = octree.query((0, 0, 0)).expect("Electron couplet at origin should exist");
        assert_eq!(c.id, 1); // Vacuum/Sink

        let expected_dna = Tpes::new(1, 0, 1, 1);
        assert_eq!(c.particle_dna, expected_dna);
        assert_eq!(c.particle_dna.net_charge(), -1);
    }
}
