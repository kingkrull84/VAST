use crate::lattice::couplet::Couplet;
use crate::lattice::octree::Octree;
use crate::tpes::Tpes;

/// Stamps a Proton triad (T:2, P:2, E:1, S:1) into a local cluster inside the Octree.
/// Injects 3 adjacent nodes forming a triangular triad:
/// - Two ID 0 Extruders at (origin + (0,0,0)) and (origin + (1,0,0))
/// - One ID 1 Vacuum at (origin + (0,1,0))
pub fn stamp_proton_triad(octree: &mut Octree, origin: (i64, i64, i64)) {
    let proton_dna = Tpes::new(2, 2, 1, 1);

    let nodes = [
        ((origin.0, origin.1, origin.2), 0u64),       // ID 0 Extruder at (0,0,0)
        ((origin.0 + 1, origin.1, origin.2), 0u64),   // ID 0 Extruder at (1,0,0)
        ((origin.0, origin.1 + 1, origin.2), 1u64),   // ID 1 Vacuum at (0,1,0)
    ];

    for (pos, id) in nodes {
        let couplet = Couplet::new(id, proton_dna);
        octree.insert(pos, couplet);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::octree::AABB;

    #[test]
    fn test_stamp_proton_triad() {
        let bounds = AABB::new(-10, 10, -10, 10, -10, 10);
        let mut octree = Octree::new(bounds, 4, 4);

        let origin = (0, 0, 0);
        stamp_proton_triad(&mut octree, origin);

        let t = octree.telemetry();
        assert_eq!(t.total_couplets, 3);

        let c0 = octree.query((0, 0, 0)).expect("Couplet at (0,0,0) should exist");
        let c1 = octree.query((1, 0, 0)).expect("Couplet at (1,0,0) should exist");
        let c2 = octree.query((0, 1, 0)).expect("Couplet at (0,1,0) should exist");

        assert_eq!(c0.id, 0); // Extruder
        assert_eq!(c1.id, 0); // Extruder
        assert_eq!(c2.id, 1); // Vacuum

        let expected_dna = Tpes::new(2, 2, 1, 1);
        assert_eq!(c0.particle_dna, expected_dna);
        assert_eq!(c1.particle_dna, expected_dna);
        assert_eq!(c2.particle_dna, expected_dna);
    }
}
