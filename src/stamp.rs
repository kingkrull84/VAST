use crate::lattice::couplet::Couplet;
use crate::lattice::octree::Octree;
use crate::tpes::Tpes;

/// Stamps a Proton triad (T:2, P:2, E:1, S:1) into a local cluster of ID 2 couplets inside the Octree.
/// Injects 3 couplets forming a triangular triad layout around the origin coordinate.
pub fn stamp_proton_triad(octree: &mut Octree, origin: (i64, i64, i64)) {
    let proton_dna = Tpes::new(2, 2, 1, 1);

    // Triad spatial offset geometry around the origin
    let offsets = [(0, 0, 0), (1, 0, 0), (0, 1, 0)];

    for offset in offsets.iter() {
        let pos = (origin.0 + offset.0, origin.1 + offset.1, origin.2 + offset.2);
        let couplet = Couplet::new_baseline(proton_dna);
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
        assert_eq!(t.total_net_charge, 3); // 3 * (2 - 1) = 3

        let c0 = octree.query((0, 0, 0));
        let c1 = octree.query((1, 0, 0));
        let c2 = octree.query((0, 1, 0));

        assert!(c0.is_some());
        assert!(c1.is_some());
        assert!(c2.is_some());

        let expected_dna = Tpes::new(2, 2, 1, 1);
        assert_eq!(c0.unwrap().particle_dna, expected_dna);
    }
}
