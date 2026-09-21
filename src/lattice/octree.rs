use crate::lattice::couplet::Couplet;
use crate::uss::Z9;
use std::collections::HashMap;

/// 3D Spatial Bounding Box in integer coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AABB {
    pub min_x: i64,
    pub max_x: i64,
    pub min_y: i64,
    pub max_y: i64,
    pub min_z: i64,
    pub max_z: i64,
}

impl AABB {
    pub fn new(min_x: i64, max_x: i64, min_y: i64, max_y: i64, min_z: i64, max_z: i64) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
        }
    }

    pub fn contains(&self, point: (i64, i64, i64)) -> bool {
        point.0 >= self.min_x
            && point.0 <= self.max_x
            && point.1 >= self.min_y
            && point.1 <= self.max_y
            && point.2 >= self.min_z
            && point.2 <= self.max_z
    }
}

/// Telemetry metrics representing the lattice stability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StabilityTelemetry {
    pub total_couplets: usize,
    pub active_nodes: usize,
    pub max_depth: usize,
    pub total_net_charge: i32,
    pub total_net_exhaust: i32,
    pub aggregate_energy: u8, // Z9 ring sum of energy
    pub aggregate_flux: u8,   // Z9 ring sum of flux
}

enum OctreeNode {
    Leaf {
        couplets: Vec<((i64, i64, i64), Couplet)>,
    },
    Internal {
        children: Box<[OctreeNode; 8]>,
    },
}

/// Sparse Octree for adaptive spatial resolution, coarse in empty space
/// and dynamically subdividing 2x2x2 down to the Planck scale where energy exists.
pub struct Octree {
    bounds: AABB,
    capacity: usize,
    max_depth: usize,
    root: OctreeNode,
}

impl Octree {
    pub fn new(bounds: AABB, capacity: usize, max_depth: usize) -> Self {
        Self {
            bounds,
            capacity,
            max_depth,
            root: OctreeNode::Leaf { couplets: Vec::new() },
        }
    }

    pub fn insert(&mut self, pos: (i64, i64, i64), couplet: Couplet) -> bool {
        if !self.bounds.contains(pos) {
            return false;
        }
        Self::insert_node(&mut self.root, self.bounds, pos, couplet, self.capacity, self.max_depth, 0)
    }

    fn insert_node(
        node: &mut OctreeNode,
        bounds: AABB,
        pos: (i64, i64, i64),
        couplet: Couplet,
        capacity: usize,
        max_depth: usize,
        current_depth: usize,
    ) -> bool {
        match node {
            OctreeNode::Leaf { couplets } => {
                for (p, c) in couplets.iter_mut() {
                    if *p == pos {
                        *c = couplet;
                        return true;
                    }
                }

                if couplets.len() < capacity || current_depth >= max_depth {
                    couplets.push((pos, couplet));
                    true
                } else {
                    let mut existing = std::mem::take(couplets);
                    existing.push((pos, couplet));

                    let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
                    let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
                    let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

                    let mut children = Box::new([
                        OctreeNode::Leaf { couplets: Vec::new() },
                        OctreeNode::Leaf { couplets: Vec::new() },
                        OctreeNode::Leaf { couplets: Vec::new() },
                        OctreeNode::Leaf { couplets: Vec::new() },
                        OctreeNode::Leaf { couplets: Vec::new() },
                        OctreeNode::Leaf { couplets: Vec::new() },
                        OctreeNode::Leaf { couplets: Vec::new() },
                        OctreeNode::Leaf { couplets: Vec::new() },
                    ]);

                    for (p, c) in existing {
                        let child_idx = Self::get_child_index(p, mid_x, mid_y, mid_z);
                        let child_bounds = Self::get_child_bounds(bounds, child_idx, mid_x, mid_y, mid_z);
                        Self::insert_node(
                            &mut children[child_idx],
                            child_bounds,
                            p,
                            c,
                            capacity,
                            max_depth,
                            current_depth + 1,
                        );
                    }

                    *node = OctreeNode::Internal { children };
                    true
                }
            }
            OctreeNode::Internal { children } => {
                let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
                let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
                let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

                let child_idx = Self::get_child_index(pos, mid_x, mid_y, mid_z);
                let child_bounds = Self::get_child_bounds(bounds, child_idx, mid_x, mid_y, mid_z);
                Self::insert_node(
                    &mut children[child_idx],
                    child_bounds,
                    pos,
                    couplet,
                    capacity,
                    max_depth,
                    current_depth + 1,
                )
            }
        }
    }

    fn get_child_index(pos: (i64, i64, i64), mid_x: i64, mid_y: i64, mid_z: i64) -> usize {
        let bit_x = if pos.0 > mid_x { 1 } else { 0 };
        let bit_y = if pos.1 > mid_y { 1 } else { 0 };
        let bit_z = if pos.2 > mid_z { 1 } else { 0 };
        (bit_x << 2) | (bit_y << 1) | bit_z
    }

    fn get_child_bounds(parent: AABB, index: usize, mid_x: i64, mid_y: i64, mid_z: i64) -> AABB {
        let min_x = if (index & 4) != 0 { mid_x + 1 } else { parent.min_x };
        let max_x = if (index & 4) != 0 { parent.max_x } else { mid_x };

        let min_y = if (index & 2) != 0 { mid_y + 1 } else { parent.min_y };
        let max_y = if (index & 2) != 0 { parent.max_y } else { mid_y };

        let min_z = if (index & 1) != 0 { mid_z + 1 } else { parent.min_z };
        let max_z = if (index & 1) != 0 { parent.max_z } else { mid_z };

        AABB::new(min_x, max_x, min_y, max_y, min_z, max_z)
    }

    pub fn query(&self, pos: (i64, i64, i64)) -> Option<&Couplet> {
        Self::query_node(&self.root, self.bounds, pos)
    }

    fn query_node<'a>(node: &'a OctreeNode, bounds: AABB, pos: (i64, i64, i64)) -> Option<&'a Couplet> {
        if !bounds.contains(pos) {
            return None;
        }
        match node {
            OctreeNode::Leaf { couplets } => {
                for (p, c) in couplets {
                    if *p == pos {
                        return Some(c);
                    }
                }
                None
            }
            OctreeNode::Internal { children } => {
                let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
                let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
                let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

                let child_idx = Self::get_child_index(pos, mid_x, mid_y, mid_z);
                let child_bounds = Self::get_child_bounds(bounds, child_idx, mid_x, mid_y, mid_z);
                Self::query_node(&children[child_idx], child_bounds, pos)
            }
        }
    }

    /// Collects all active couplet positions and pressures into a hash map.
    fn collect_pressures(node: &OctreeNode, map: &mut HashMap<(i64, i64, i64), Z9>) {
        match node {
            OctreeNode::Leaf { couplets } => {
                for (pos, couplet) in couplets {
                    map.insert(*pos, couplet.pressure);
                }
            }
            OctreeNode::Internal { children } => {
                for child in children.iter() {
                    Self::collect_pressures(child, map);
                }
            }
        }
    }

    /// Steps simulation by updating flux state of all couplets inside the octree using Z/9Z arithmetic,
    /// incorporating 6-neighbor stencil lookup for adjacent pressure transfer.
    pub fn step(&mut self) {
        let mut pressure_map = HashMap::new();
        Self::collect_pressures(&self.root, &mut pressure_map);
        Self::step_node(&mut self.root, &pressure_map);
    }

    fn step_node(node: &mut OctreeNode, pressure_map: &HashMap<(i64, i64, i64), Z9>) {
        match node {
            OctreeNode::Leaf { couplets } => {
                let neighbor_offsets: [(i64, i64, i64); 6] = [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ];

                for (pos, couplet) in couplets {
                    // Accumulate pressure from 6 adjacent neighbors in Z/9Z ring
                    let mut neighbor_pressure = Z9::ZERO;
                    for offset in &neighbor_offsets {
                        let n_pos = (pos.0 + offset.0, pos.1 + offset.1, pos.2 + offset.2);
                        if let Some(&p) = pressure_map.get(&n_pos) {
                            neighbor_pressure += p;
                        }
                    }

                    // Update couplet flux with local energy and combined (local + neighbor) pressure
                    let total_pressure = couplet.pressure + neighbor_pressure;
                    couplet.flux = Z9::update_flux(couplet.energy, total_pressure);
                    couplet.pressure += Z9::ONE;
                }
            }
            OctreeNode::Internal { children } => {
                for child in children.iter_mut() {
                    Self::step_node(child, pressure_map);
                }
            }
        }
    }

    /// Collects and computes current stability telemetry metrics.
    pub fn telemetry(&self) -> StabilityTelemetry {
        let mut total_couplets = 0;
        let mut active_nodes = 0;
        let mut max_observed_depth = 0;
        let mut total_net_charge = 0;
        let mut total_net_exhaust = 0;
        let mut aggregate_energy = Z9::ZERO;
        let mut aggregate_flux = Z9::ZERO;

        Self::collect_telemetry(
            &self.root,
            0,
            &mut total_couplets,
            &mut active_nodes,
            &mut max_observed_depth,
            &mut total_net_charge,
            &mut total_net_exhaust,
            &mut aggregate_energy,
            &mut aggregate_flux,
        );

        StabilityTelemetry {
            total_couplets,
            active_nodes,
            max_depth: max_observed_depth,
            total_net_charge,
            total_net_exhaust,
            aggregate_energy: aggregate_energy.value(),
            aggregate_flux: aggregate_flux.value(),
        }
    }

    fn collect_telemetry(
        node: &OctreeNode,
        current_depth: usize,
        total_couplets: &mut usize,
        active_nodes: &mut usize,
        max_observed_depth: &mut usize,
        total_net_charge: &mut i32,
        total_net_exhaust: &mut i32,
        aggregate_energy: &mut Z9,
        aggregate_flux: &mut Z9,
    ) {
        *active_nodes += 1;
        if current_depth > *max_observed_depth {
            *max_observed_depth = current_depth;
        }

        match node {
            OctreeNode::Leaf { couplets } => {
                *total_couplets += couplets.len();
                for (_, couplet) in couplets {
                    *total_net_charge += couplet.particle_dna.net_charge();
                    *total_net_exhaust += couplet.particle_dna.net_exhaust();
                    *aggregate_energy += couplet.energy;
                    *aggregate_flux += couplet.flux;
                }
            }
            OctreeNode::Internal { children } => {
                for child in children.iter() {
                    Self::collect_telemetry(
                        child,
                        current_depth + 1,
                        total_couplets,
                        active_nodes,
                        max_observed_depth,
                        total_net_charge,
                        total_net_exhaust,
                        aggregate_energy,
                        aggregate_flux,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tpes::Tpes;

    #[test]
    fn test_octree_insertion_and_query() {
        let bounds = AABB::new(-10, 10, -10, 10, -10, 10);
        let mut octree = Octree::new(bounds, 1, 4);

        let dna = Tpes::new(2, 2, 1, 1);
        let couplet = Couplet::new_baseline(dna);

        assert!(octree.insert((0, 0, 0), couplet.clone()));
        let queried = octree.query((0, 0, 0));
        assert!(queried.is_some());
        assert_eq!(queried.unwrap(), &couplet);
    }

    #[test]
    fn test_octree_subdivision() {
        let bounds = AABB::new(-10, 10, -10, 10, -10, 10);
        let mut octree = Octree::new(bounds, 1, 4); // Capacity 1 forces subdivision on 2nd insertion

        let dna1 = Tpes::new(2, 2, 1, 1);
        let dna2 = Tpes::new(1, 1, 0, 0);

        octree.insert((1, 1, 1), Couplet::new_baseline(dna1));
        octree.insert((-1, -1, -1), Couplet::new_baseline(dna2));

        let t = octree.telemetry();
        assert_eq!(t.total_couplets, 2);
        assert!(t.max_depth > 0);
    }

    #[test]
    fn test_octree_step_and_telemetry() {
        let bounds = AABB::new(-10, 10, -10, 10, -10, 10);
        let mut octree = Octree::new(bounds, 4, 4);

        let proton_dna = Tpes::new(2, 2, 1, 1);
        octree.insert((0, 0, 0), Couplet::new_baseline(proton_dna));

        let t1 = octree.telemetry();
        assert_eq!(t1.aggregate_energy, 1);
        assert_eq!(t1.aggregate_flux, 0);

        octree.step();

        let t2 = octree.telemetry();
        assert_eq!(t2.aggregate_flux, 1);
    }

    #[test]
    fn test_6_neighbor_stencil_flux_transfer() {
        let bounds = AABB::new(-10, 10, -10, 10, -10, 10);
        let mut octree = Octree::new(bounds, 10, 4);

        let dna = Tpes::new(2, 2, 1, 1);
        let mut c1 = Couplet::new_baseline(dna);
        let c2 = Couplet::new_baseline(dna);

        c1.pressure = Z9::new(3); // set c1 pressure to 3

        octree.insert((0, 0, 0), c1);
        octree.insert((1, 0, 0), c2); // adjacent along x-axis

        // Before step, c2 has energy=1, pressure=0
        // During step, c2 perceives neighbor pressure 3 from c1 at (0,0,0)
        // total_pressure for c2 = 0 + 3 = 3
        // flux for c2 = (1 + 3 * 2) mod 9 = 7 mod 9 = 7
        octree.step();

        let c2_updated = octree.query((1, 0, 0)).unwrap();
        assert_eq!(c2_updated.flux, Z9::new(7));
    }
}
