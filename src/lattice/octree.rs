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

#[derive(Debug, Clone)]
#[repr(align(64))]
pub enum OctreeNode {
    Leaf {
        couplets: Vec<((i64, i64, i64), Couplet)>,
    },
    Internal {
        children_indices: [usize; 8],
    },
}

/// Sparse Octree utilizing a contiguous 1D memory pool (arena) and strict double-buffering
/// (`read_state` and `write_state`) for 100% deterministic, lock-free parallelizable simulation steps.
pub struct Octree {
    bounds: AABB,
    capacity: usize,
    max_depth: usize,
    read_state: Vec<OctreeNode>,
    write_state: Vec<OctreeNode>,
}

impl Octree {
    pub fn new(bounds: AABB, capacity: usize, max_depth: usize) -> Self {
        let root = OctreeNode::Leaf { couplets: Vec::new() };
        Self {
            bounds,
            capacity,
            max_depth,
            read_state: vec![root.clone()],
            write_state: vec![root],
        }
    }

    pub fn insert(&mut self, pos: (i64, i64, i64), couplet: Couplet) -> bool {
        if !self.bounds.contains(pos) {
            return false;
        }
        let bounds = self.bounds;
        let cap = self.capacity;
        let max_d = self.max_depth;

        let inserted = Self::insert_into_pool(&mut self.read_state, 0, bounds, pos, couplet, cap, max_d, 0);
        if inserted {
            // Keep write_state synchronized with read_state topology and couplet state
            self.write_state = self.read_state.clone();
        }
        inserted
    }

    fn insert_into_pool(
        pool: &mut Vec<OctreeNode>,
        node_idx: usize,
        bounds: AABB,
        pos: (i64, i64, i64),
        couplet: Couplet,
        capacity: usize,
        max_depth: usize,
        current_depth: usize,
    ) -> bool {
        let is_leaf = matches!(pool[node_idx], OctreeNode::Leaf { .. });

        if is_leaf {
            let mut subdivide = false;
            let mut existing_couplets = Vec::new();

            if let OctreeNode::Leaf { couplets } = &mut pool[node_idx] {
                for (p, c) in couplets.iter_mut() {
                    if *p == pos {
                        *c = couplet.clone();
                        return true;
                    }
                }

                if couplets.len() < capacity || current_depth >= max_depth {
                    couplets.push((pos, couplet));
                    return true;
                } else {
                    subdivide = true;
                    existing_couplets = std::mem::take(couplets);
                    existing_couplets.push((pos, couplet));
                }
            }

            if subdivide {
                let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
                let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
                let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

                let mut children_indices = [0usize; 8];
                for i in 0..8 {
                    let child_node_idx = pool.len();
                    pool.push(OctreeNode::Leaf { couplets: Vec::new() });
                    children_indices[i] = child_node_idx;
                }

                pool[node_idx] = OctreeNode::Internal { children_indices };

                for (p, c) in existing_couplets {
                    let child_slot = Self::get_child_index(p, mid_x, mid_y, mid_z);
                    let child_idx = children_indices[child_slot];
                    let child_bounds = Self::get_child_bounds(bounds, child_slot, mid_x, mid_y, mid_z);
                    Self::insert_into_pool(
                        pool,
                        child_idx,
                        child_bounds,
                        p,
                        c,
                        capacity,
                        max_depth,
                        current_depth + 1,
                    );
                }
                return true;
            }
        } else {
            let children_indices = match &pool[node_idx] {
                OctreeNode::Internal { children_indices } => *children_indices,
                _ => unreachable!(),
            };

            let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
            let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
            let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

            let child_slot = Self::get_child_index(pos, mid_x, mid_y, mid_z);
            let child_idx = children_indices[child_slot];
            let child_bounds = Self::get_child_bounds(bounds, child_slot, mid_x, mid_y, mid_z);
            return Self::insert_into_pool(
                pool,
                child_idx,
                child_bounds,
                pos,
                couplet,
                capacity,
                max_depth,
                current_depth + 1,
            );
        }

        false
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
        Self::query_pool(&self.read_state, 0, self.bounds, pos)
    }

    fn query_pool<'a>(
        pool: &'a [OctreeNode],
        node_idx: usize,
        bounds: AABB,
        pos: (i64, i64, i64),
    ) -> Option<&'a Couplet> {
        if !bounds.contains(pos) || node_idx >= pool.len() {
            return None;
        }

        match &pool[node_idx] {
            OctreeNode::Leaf { couplets } => {
                for (p, c) in couplets {
                    if *p == pos {
                        return Some(c);
                    }
                }
                None
            }
            OctreeNode::Internal { children_indices } => {
                let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
                let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
                let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

                let child_slot = Self::get_child_index(pos, mid_x, mid_y, mid_z);
                let child_idx = children_indices[child_slot];
                let child_bounds = Self::get_child_bounds(bounds, child_slot, mid_x, mid_y, mid_z);
                Self::query_pool(pool, child_idx, child_bounds, pos)
            }
        }
    }

    /// Collects all active couplet positions and pressures into a hash map from the read pool.
    fn collect_pressures_from_pool(pool: &[OctreeNode], node_idx: usize, map: &mut HashMap<(i64, i64, i64), Z9>) {
        if node_idx >= pool.len() {
            return;
        }
        match &pool[node_idx] {
            OctreeNode::Leaf { couplets } => {
                for (pos, couplet) in couplets {
                    map.insert(*pos, couplet.pressure);
                }
            }
            OctreeNode::Internal { children_indices } => {
                for &child_idx in children_indices {
                    Self::collect_pressures_from_pool(pool, child_idx, map);
                }
            }
        }
    }

    /// Steps simulation strictly using double-buffering.
    /// Reads current state from `read_state`, computes 6-neighbor flux updates into `write_state`,
    /// and performs `std::mem::swap(&mut self.read_state, &mut self.write_state)` at the end of the tick.
    pub fn step(&mut self) {
        let mut pressure_map = HashMap::new();
        Self::collect_pressures_from_pool(&self.read_state, 0, &mut pressure_map);

        let neighbor_offsets: [(i64, i64, i64); 6] = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];

        // Ensure write_state matches topology of read_state
        if self.write_state.len() != self.read_state.len() {
            self.write_state = self.read_state.clone();
        }

        for idx in 0..self.read_state.len() {
            if let (OctreeNode::Leaf { couplets: read_couplets }, OctreeNode::Leaf { couplets: write_couplets }) =
                (&self.read_state[idx], &mut self.write_state[idx])
            {
                for (i, (pos, read_couplet)) in read_couplets.iter().enumerate() {
                    let mut neighbor_pressure = Z9::ZERO;
                    for offset in &neighbor_offsets {
                        let n_pos = (pos.0 + offset.0, pos.1 + offset.1, pos.2 + offset.2);
                        if let Some(&p) = pressure_map.get(&n_pos) {
                            neighbor_pressure += p;
                        }
                    }

                    let total_pressure = read_couplet.pressure + neighbor_pressure;
                    let next_flux = Z9::update_flux(read_couplet.energy, total_pressure);
                    let next_pressure = read_couplet.pressure + Z9::ONE;

                    write_couplets[i].1.flux = next_flux;
                    write_couplets[i].1.pressure = next_pressure;
                }
            }
        }

        // Fast double-buffering pool swap
        std::mem::swap(&mut self.read_state, &mut self.write_state);
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

        Self::collect_telemetry_from_pool(
            &self.read_state,
            0,
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

    fn collect_telemetry_from_pool(
        pool: &[OctreeNode],
        node_idx: usize,
        current_depth: usize,
        total_couplets: &mut usize,
        active_nodes: &mut usize,
        max_observed_depth: &mut usize,
        total_net_charge: &mut i32,
        total_net_exhaust: &mut i32,
        aggregate_energy: &mut Z9,
        aggregate_flux: &mut Z9,
    ) {
        if node_idx >= pool.len() {
            return;
        }

        *active_nodes += 1;
        if current_depth > *max_observed_depth {
            *max_observed_depth = current_depth;
        }

        match &pool[node_idx] {
            OctreeNode::Leaf { couplets } => {
                *total_couplets += couplets.len();
                for (_, couplet) in couplets {
                    *total_net_charge += couplet.particle_dna.net_charge();
                    *total_net_exhaust += couplet.particle_dna.net_exhaust();
                    *aggregate_energy += couplet.energy;
                    *aggregate_flux += couplet.flux;
                }
            }
            OctreeNode::Internal { children_indices } => {
                for &child_idx in children_indices {
                    Self::collect_telemetry_from_pool(
                        pool,
                        child_idx,
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
    fn test_octree_node_alignment() {
        assert_eq!(std::mem::align_of::<OctreeNode>(), 64);
    }

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
        let mut octree = Octree::new(bounds, 1, 4);

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

        c1.pressure = Z9::new(3);

        octree.insert((0, 0, 0), c1);
        octree.insert((1, 0, 0), c2);

        octree.step();

        let c2_updated = octree.query((1, 0, 0)).unwrap();
        assert_eq!(c2_updated.flux, Z9::new(7));
    }
}
