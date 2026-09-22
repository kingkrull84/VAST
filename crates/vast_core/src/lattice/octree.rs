use crate::lattice::couplet::Couplet;
use crate::uss::Z9;
use rayon::prelude::*;
use std::collections::HashMap;

/// D3Q27 discrete velocity directional vectors (dx, dy, dz) where dx, dy, dz in {-1, 0, 1}.
pub const DIRECTION_VECTORS: [(i64, i64, i64); 27] = [
    (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
    (-1,  0, -1), (-1,  0, 0), (-1,  0, 1),
    (-1,  1, -1), (-1,  1, 0), (-1,  1, 1),
    ( 0, -1, -1), ( 0, -1, 0), ( 0, -1, 1),
    ( 0,  0, -1), ( 0,  0, 0), ( 0,  0, 1),
    ( 0,  1, -1), ( 0,  1, 0), ( 0,  1, 1),
    ( 1, -1, -1), ( 1, -1, 0), ( 1, -1, 1),
    ( 1,  0, -1), ( 1,  0, 0), ( 1,  0, 1),
    ( 1,  1, -1), ( 1,  1, 0), ( 1,  1, 1),
];

/// Converts 3D discrete direction vector (dx, dy, dz) to D3Q27 array index in 0..27.
pub fn direction_to_index(dx: i64, dy: i64, dz: i64) -> usize {
    ((dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)) as usize
}

/// Computes a pair of orthogonal direction indices (o1, o2) for a given direction index k.
pub fn get_orthogonal_pair(k: usize) -> (usize, usize) {
    let dir = DIRECTION_VECTORS[k];
    let dx = dir.0;
    let dy = dir.1;
    let dz = dir.2;

    let o1_vec = if dx != 0 || dy != 0 {
        (-dy, dx, 0)
    } else {
        (1, 0, 0)
    };

    let v2 = (
        dy * o1_vec.2 - dz * o1_vec.1,
        dz * o1_vec.0 - dx * o1_vec.2,
        dx * o1_vec.1 - dy * o1_vec.0,
    );

    let o2_vec = (-v2.0.signum(), -v2.1.signum(), -v2.2.signum());

    (
        direction_to_index(o1_vec.0, o1_vec.1, o1_vec.2),
        direction_to_index(o2_vec.0, o2_vec.1, o2_vec.2),
    )
}

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

    /// Splits this AABB into 8 equal sub-octants (2x2x2 grid).
    pub fn subdivide(&self) -> [AABB; 8] {
        let mid_x = self.min_x + (self.max_x - self.min_x) / 2;
        let mid_y = self.min_y + (self.max_y - self.min_y) / 2;
        let mid_z = self.min_z + (self.max_z - self.min_z) / 2;

        let mut sub_octants = [AABB::new(0, 0, 0, 0, 0, 0); 8];
        for i in 0..8 {
            let min_x = if (i & 4) != 0 { mid_x + 1 } else { self.min_x };
            let max_x = if (i & 4) != 0 { self.max_x } else { mid_x };

            let min_y = if (i & 2) != 0 { mid_y + 1 } else { self.min_y };
            let max_y = if (i & 2) != 0 { self.max_y } else { mid_y };

            let min_z = if (i & 1) != 0 { mid_z + 1 } else { self.min_z };
            let max_z = if (i & 1) != 0 { self.max_z } else { mid_z };

            sub_octants[i] = AABB::new(min_x, max_x, min_y, max_y, min_z, max_z);
        }
        sub_octants
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
    Branch {
        children_indices: [usize; 8],
    },
}

impl OctreeNode {
    /// When called on a Leaf, converts it into a Branch containing 8 default child Leaf nodes
    /// in the arena pool (representing the sub-octants).
    pub fn subdivide(&mut self, pool: &mut Vec<OctreeNode>) -> [usize; 8] {
        match self {
            OctreeNode::Leaf { couplets } => {
                let _old_couplets = std::mem::take(couplets);
                let mut children_indices = [0usize; 8];
                for i in 0..8 {
                    let child_idx = pool.len();
                    pool.push(OctreeNode::Leaf {
                        couplets: Vec::new(),
                    });
                    children_indices[i] = child_idx;
                }
                *self = OctreeNode::Branch { children_indices };
                children_indices
            }
            OctreeNode::Branch { children_indices } => *children_indices,
        }
    }

    /// Static helper to subdivide a node at a given index in the arena pool.
    pub fn subdivide_at(pool: &mut Vec<OctreeNode>, node_idx: usize) -> [usize; 8] {
        if matches!(pool.get(node_idx), Some(OctreeNode::Leaf { .. })) {
            let mut children_indices = [0usize; 8];
            for i in 0..8 {
                let child_idx = pool.len();
                pool.push(OctreeNode::Leaf {
                    couplets: Vec::new(),
                });
                children_indices[i] = child_idx;
            }
            if let OctreeNode::Leaf { couplets } = &mut pool[node_idx] {
                let _ = std::mem::take(couplets);
            }
            pool[node_idx] = OctreeNode::Branch { children_indices };
            children_indices
        } else if let Some(OctreeNode::Branch { children_indices }) = pool.get(node_idx) {
            *children_indices
        } else {
            unreachable!()
        }
    }

    /// When called on a Branch where all 8 children are Leaves in a uniform resting state (flux = 0),
    /// merges them back into a single parent Leaf.
    pub fn collapse(&mut self, pool: &[OctreeNode]) -> bool {
        if let OctreeNode::Branch { children_indices } = *self {
            for &child_idx in &children_indices {
                if child_idx >= pool.len() {
                    return false;
                }
                match &pool[child_idx] {
                    OctreeNode::Leaf { couplets } => {
                        for (_, couplet) in couplets {
                            if couplet.total_flux().value() != 0 {
                                return false;
                            }
                        }
                    }
                    OctreeNode::Branch { .. } => return false,
                }
            }

            let mut merged_couplets = Vec::new();
            for &child_idx in &children_indices {
                if let OctreeNode::Leaf { couplets } = &pool[child_idx] {
                    merged_couplets.extend(couplets.clone());
                }
            }

            *self = OctreeNode::Leaf {
                couplets: merged_couplets,
            };
            true
        } else {
            false
        }
    }

    /// Static helper to collapse a node at a given index in the arena pool.
    pub fn collapse_at(pool: &mut Vec<OctreeNode>, node_idx: usize) -> bool {
        Self::collapse_at_with_depth(pool, node_idx, 0, 8)
    }

    /// Helper to collapse a node while preserving Geometric Gravity resolution depths.
    pub fn collapse_at_with_depth(
        pool: &mut Vec<OctreeNode>,
        node_idx: usize,
        current_depth: usize,
        max_depth: usize,
    ) -> bool {
        let children_indices = match pool.get(node_idx) {
            Some(OctreeNode::Branch { children_indices }) => *children_indices,
            _ => return false,
        };

        for &child_idx in &children_indices {
            if child_idx >= pool.len() {
                return false;
            }
            match &pool[child_idx] {
                OctreeNode::Leaf { couplets } => {
                    for (pos, couplet) in couplets {
                        if couplet.total_flux().value() != 0 {
                            return false;
                        }
                        if couplet.id == 2 {
                            let dist = pos.0.abs().max(pos.1.abs()).max(pos.2.abs());
                            let target_depth = (9 - dist).clamp(1, max_depth as i64) as usize;
                            if current_depth < target_depth {
                                return false;
                            }
                        }
                    }
                }
                OctreeNode::Branch { .. } => return false,
            }
        }

        let mut merged_couplets = Vec::new();
        for &child_idx in &children_indices {
            if let OctreeNode::Leaf { couplets } = &pool[child_idx] {
                merged_couplets.extend(couplets.clone());
            }
        }

        pool[node_idx] = OctreeNode::Leaf {
            couplets: merged_couplets,
        };
        true
    }
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

                let children_indices = OctreeNode::subdivide_at(pool, node_idx);

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
                OctreeNode::Branch { children_indices } => *children_indices,
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
            OctreeNode::Branch { children_indices } => {
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

    pub fn delete(&mut self, pos: (i64, i64, i64)) -> bool {
        if !self.bounds.contains(pos) {
            return false;
        }
        let deleted = Self::delete_from_pool(&mut self.read_state, 0, self.bounds, pos);
        if deleted {
            self.write_state = self.read_state.clone();
        }
        deleted
    }

    fn delete_from_pool(
        pool: &mut Vec<OctreeNode>,
        node_idx: usize,
        bounds: AABB,
        pos: (i64, i64, i64),
    ) -> bool {
        if !bounds.contains(pos) || node_idx >= pool.len() {
            return false;
        }

        match &mut pool[node_idx] {
            OctreeNode::Leaf { couplets } => {
                if let Some(idx) = couplets.iter().position(|(p, _)| *p == pos) {
                    couplets.swap_remove(idx);
                    true
                } else {
                    false
                }
            }
            OctreeNode::Branch { children_indices } => {
                let children_indices = *children_indices;
                let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
                let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
                let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

                let child_slot = Self::get_child_index(pos, mid_x, mid_y, mid_z);
                let child_idx = children_indices[child_slot];
                let child_bounds = Self::get_child_bounds(bounds, child_slot, mid_x, mid_y, mid_z);
                Self::delete_from_pool(pool, child_idx, child_bounds, pos)
            }
        }
    }

    /// Collects all active couplets into a hash map from the read pool.
    fn collect_couplets_from_pool(pool: &[OctreeNode], node_idx: usize, map: &mut HashMap<(i64, i64, i64), Couplet>) {
        if node_idx >= pool.len() {
            return;
        }
        match &pool[node_idx] {
            OctreeNode::Leaf { couplets } => {
                for (pos, couplet) in couplets {
                    map.insert(*pos, couplet.clone());
                }
            }
            OctreeNode::Branch { children_indices } => {
                for &child_idx in children_indices {
                    Self::collect_couplets_from_pool(pool, child_idx, map);
                }
            }
        }
    }

    /// Steps simulation strictly using double-buffering and D3Q27 Lattice Boltzmann dynamics.
    /// Reads current state from `read_state`, performs directional advection, topological sink pulling,
    /// and orthogonal scattering into `write_state`, then evaluates dynamic subdivision/collapse.
    pub fn step(&mut self, threshold: u8) {
        let mut read_map = HashMap::new();
        Self::collect_couplets_from_pool(&self.read_state, 0, &mut read_map);

        // Pre-step: Identity Routing & Monopole Drain (Distance 1 unbinding)
        let mut consumed_positions = Vec::new();
        let mut recycled_boundary_positions = Vec::new();
        let mut id1_absorptions: HashMap<(i64, i64, i64), (u32, [u32; 27])> = HashMap::new();

        for (&id1_pos, id1_couplet) in &read_map {
            if id1_couplet.id == 1 || id1_couplet.particle_dna.net_charge() < 0 {
                for k in 0..27 {
                    if k == 13 {
                        continue;
                    }
                    let dir = DIRECTION_VECTORS[k];
                    let neighbor_pos = (id1_pos.0 + dir.0, id1_pos.1 + dir.1, id1_pos.2 + dir.2);
                    if let Some(neighbor) = read_map.get(&neighbor_pos) {
                        if neighbor.id == 2 {
                            consumed_positions.push(neighbor_pos);

                            let bx = if dir.0 > 0 { self.bounds.max_x } else if dir.0 < 0 { self.bounds.min_x } else { 0 };
                            let by = if dir.1 > 0 { self.bounds.max_y } else if dir.1 < 0 { self.bounds.min_y } else { 0 };
                            let bz = if dir.2 > 0 { self.bounds.max_z } else if dir.2 < 0 { self.bounds.min_z } else { 0 };
                            let boundary_pos = if bx != 0 || by != 0 || bz != 0 {
                                (bx, by, bz)
                            } else {
                                (self.bounds.max_x, 0, 0)
                            };
                            recycled_boundary_positions.push(boundary_pos);

                            let entry = id1_absorptions.entry(id1_pos).or_insert((0, [0; 27]));
                            entry.0 += 1;
                            let b_indices = [22, 4, 16, 10]; // +X, -X, +Y, -Y
                            for &idx in &b_indices {
                                entry.1[idx] += 1;
                            }
                        }
                    }
                }
            }
        }

        // Delete consumed ID 2 couplets from the octree
        for pos in &consumed_positions {
            Self::delete_from_pool(&mut self.read_state, 0, self.bounds, *pos);
            read_map.remove(pos);
        }

        // Toroidal Boundary Recycling: inject new ID 2 couplets at outer boundary shell
        let element_zero_dna = crate::tpes::Tpes::new(2, 2, 1, 1);
        for boundary_pos in recycled_boundary_positions {
            let cap = self.capacity;
            let max_d = self.max_depth;
            let bounds = self.bounds;
            let couplet = Couplet::new_baseline(element_zero_dna);
            Self::insert_into_pool(&mut self.read_state, 0, bounds, boundary_pos, couplet, cap, max_d, 0);
        }

        // Apply energy absorption & tangential flux ejections to ID 1 nodes in read_map
        for (id1_pos, (add_energy, ejections)) in &id1_absorptions {
            if let Some(c) = read_map.get_mut(id1_pos) {
                c.energy += Z9::new(*add_energy as i32);
                for k in 0..27 {
                    c.flux[k] += Z9::new(ejections[k] as i32);
                }
            }
        }

        // Synchronize write_state with read_state following deletions and insertions
        self.write_state = self.read_state.clone();

        self.read_state
            .par_iter()
            .zip(self.write_state.par_iter_mut())
            .for_each(|(read_node, write_node)| {
                if let (
                    OctreeNode::Leaf { couplets: read_couplets },
                    OctreeNode::Leaf { couplets: write_couplets },
                ) = (read_node, write_node)
                {
                    for (i, (pos, read_couplet)) in read_couplets.iter().enumerate() {
                        let mut advected_flux = [Z9::ZERO; 27];

                        // Step 1: Directional Advection
                        for k in 0..27 {
                            let dir = DIRECTION_VECTORS[k];
                            let source_pos = (pos.0 - dir.0, pos.1 - dir.1, pos.2 - dir.2);
                            if let Some(source_couplet) = read_map.get(&source_pos) {
                                advected_flux[k] = source_couplet.flux[k];
                            }
                        }

                        // Include tangential momentum ejections from pre-step monopole drain unbinding
                        if let Some((_, ejections)) = id1_absorptions.get(pos) {
                            for k in 0..27 {
                                if ejections[k] > 0 {
                                    advected_flux[k] += Z9::new(ejections[k] as i32);
                                }
                            }
                        }

                        // Step 2: Topological Sink (Electron) vs Base Node Flux
                        if read_couplet.particle_dna.net_charge() < 0 || read_couplet.id == 1 {
                            // Electron Topological Sink: pull flux from poles (+Z: k=14, -Z: k=12)
                            let polar_flux = advected_flux[14] + advected_flux[12];
                            advected_flux[14] = Z9::ZERO;
                            advected_flux[12] = Z9::ZERO;

                            let base_flux = Z9::update_flux(read_couplet.energy, read_couplet.pressure);
                            let total_scatter = polar_flux + base_flux;

                            let val = total_scatter.value();
                            let q = val / 4;
                            let r = val % 4;
                            let b_indices = [22, 4, 16, 10]; // +X, -X, +Y, -Y
                            for (b_i, &idx) in b_indices.iter().enumerate() {
                                let add_amount = q + if (b_i as u8) < r { 1 } else { 0 };
                                advected_flux[idx] += Z9::new(add_amount as i32);
                            }
                        } else {
                            // Base node flux generation added to rest bucket (k=13)
                            let base_flux = Z9::update_flux(read_couplet.energy, read_couplet.pressure);
                            advected_flux[13] += base_flux;
                        }

                        // Step 3: Orthogonal Scattering for opposing flux packets
                        for k in 0..13 {
                            let k_opp = 26 - k;
                            let f1 = advected_flux[k].value();
                            let f2 = advected_flux[k_opp].value();
                            if f1 > 0 && f2 > 0 {
                                let c = f1.min(f2);
                                let c_z9 = Z9::new(c as i32);
                                advected_flux[k] -= c_z9;
                                advected_flux[k_opp] -= c_z9;
                                let (o1, o2) = get_orthogonal_pair(k);
                                advected_flux[o1] += c_z9;
                                advected_flux[o2] += c_z9;
                            }
                        }

                        write_couplets[i].1.flux = advected_flux;
                        write_couplets[i].1.pressure = read_couplet.pressure + Z9::ONE;
                        if let Some(updated_couplet) = read_map.get(pos) {
                            write_couplets[i].1.energy = updated_couplet.energy;
                        } else {
                            write_couplets[i].1.energy = read_couplet.energy;
                        }
                    }
                }
            });

        // Fast double-buffering pool swap
        std::mem::swap(&mut self.read_state, &mut self.write_state);

        // Dynamic subdivision and collapse post-processing evaluation
        Self::evaluate_dynamic_nodes(
            &mut self.read_state,
            0,
            self.bounds,
            threshold,
            self.max_depth,
            0,
        );

        self.write_state = self.read_state.clone();
    }

    fn evaluate_dynamic_nodes(
        pool: &mut Vec<OctreeNode>,
        node_idx: usize,
        bounds: AABB,
        threshold: u8,
        max_depth: usize,
        current_depth: usize,
    ) {
        if node_idx >= pool.len() {
            return;
        }

        let is_leaf = matches!(pool[node_idx], OctreeNode::Leaf { .. });

        if is_leaf {
            let mut should_subdivide = false;
            let mut couplets_to_redistribute = Vec::new();

            if let OctreeNode::Leaf { couplets } = &pool[node_idx] {
                if current_depth < max_depth {
                    for (pos, couplet) in couplets {
                        if couplet.id == 2 {
                            let dist = pos.0.abs().max(pos.1.abs()).max(pos.2.abs());
                            let target_depth = (9 - dist).clamp(1, max_depth as i64) as usize;
                            if current_depth < target_depth {
                                should_subdivide = true;
                                break;
                            }
                        }
                    }

                    if !should_subdivide && couplets.len() > 1 {
                        let max_flux = couplets.iter().map(|(_, c)| c.total_flux().value()).max().unwrap_or(0);
                        let min_flux = couplets.iter().map(|(_, c)| c.total_flux().value()).min().unwrap_or(0);
                        let gradient = max_flux.saturating_sub(min_flux);
                        if gradient > threshold {
                            should_subdivide = true;
                        }
                    }

                    if should_subdivide {
                        couplets_to_redistribute = couplets.clone();
                    }
                }
            }

            if should_subdivide {
                let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
                let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
                let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

                let child_indices = OctreeNode::subdivide_at(pool, node_idx);

                for (p, c) in couplets_to_redistribute {
                    let slot = Self::get_child_index(p, mid_x, mid_y, mid_z);
                    let child_idx = child_indices[slot];
                    if let OctreeNode::Leaf { couplets } = &mut pool[child_idx] {
                        couplets.push((p, c));
                    }
                }

                for i in 0..8 {
                    let child_idx = child_indices[i];
                    let child_bounds = Self::get_child_bounds(bounds, i, mid_x, mid_y, mid_z);
                    Self::evaluate_dynamic_nodes(
                        pool,
                        child_idx,
                        child_bounds,
                        threshold,
                        max_depth,
                        current_depth + 1,
                    );
                }
            }
        } else {
            let children_indices = match &pool[node_idx] {
                OctreeNode::Branch { children_indices } => *children_indices,
                _ => unreachable!(),
            };

            let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
            let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
            let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

            for i in 0..8 {
                let child_idx = children_indices[i];
                let child_bounds = Self::get_child_bounds(bounds, i, mid_x, mid_y, mid_z);
                Self::evaluate_dynamic_nodes(
                    pool,
                    child_idx,
                    child_bounds,
                    threshold,
                    max_depth,
                    current_depth + 1,
                );
            }

            OctreeNode::collapse_at_with_depth(pool, node_idx, current_depth, max_depth);
        }
    }

    /// Collects active leaf regions (AABBs) and active couplets for 3D visual telemetry.
    pub fn collect_active_leaves(&self) -> Vec<(AABB, Vec<((i64, i64, i64), Couplet)>)> {
        let mut leaves = Vec::new();
        Self::collect_leaves_from_pool(&self.read_state, 0, self.bounds, &mut leaves);
        leaves
    }

    fn collect_leaves_from_pool(
        pool: &[OctreeNode],
        node_idx: usize,
        bounds: AABB,
        leaves: &mut Vec<(AABB, Vec<((i64, i64, i64), Couplet)>)>,
    ) {
        if node_idx >= pool.len() {
            return;
        }
        match &pool[node_idx] {
            OctreeNode::Leaf { couplets } => {
                if !couplets.is_empty() {
                    leaves.push((bounds, couplets.clone()));
                }
            }
            OctreeNode::Branch { children_indices } => {
                let mid_x = bounds.min_x + (bounds.max_x - bounds.min_x) / 2;
                let mid_y = bounds.min_y + (bounds.max_y - bounds.min_y) / 2;
                let mid_z = bounds.min_z + (bounds.max_z - bounds.min_z) / 2;

                for (i, &child_idx) in children_indices.iter().enumerate() {
                    let child_bounds = Self::get_child_bounds(bounds, i, mid_x, mid_y, mid_z);
                    Self::collect_leaves_from_pool(pool, child_idx, child_bounds, leaves);
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
                    *aggregate_flux += couplet.total_flux();
                }
            }
            OctreeNode::Branch { children_indices } => {
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

        octree.step(2);

        let t2 = octree.telemetry();
        assert_eq!(t2.aggregate_flux, 1);
    }

    #[test]
    fn test_collect_active_leaves() {
        let bounds = AABB::new(-10, 10, -10, 10, -10, 10);
        let mut octree = Octree::new(bounds, 1, 4);

        let dna = Tpes::new(2, 2, 1, 1);
        octree.insert((1, 1, 1), Couplet::new_baseline(dna));

        let active_leaves = octree.collect_active_leaves();
        assert_eq!(active_leaves.len(), 1);
        assert_eq!(active_leaves[0].1.len(), 1);
        assert_eq!(active_leaves[0].1[0].0, (1, 1, 1));
    }

    #[test]
    fn test_directional_advection_and_scattering() {
        let bounds = AABB::new(-10, 10, -10, 10, -10, 10);
        let mut octree = Octree::new(bounds, 10, 4);

        let dna = Tpes::new(2, 2, 1, 1);
        let mut c1 = Couplet::new_baseline(dna);
        let idx_plus_x = direction_to_index(1, 0, 0);
        c1.flux[idx_plus_x] = Z9::new(5);

        octree.insert((0, 0, 0), c1);
        octree.insert((1, 0, 0), Couplet::new_baseline(dna));

        octree.step(2);

        // Flux in +X direction should advect from (0,0,0) to (1,0,0)
        let c2_updated = octree.query((1, 0, 0)).unwrap();
        assert_eq!(c2_updated.flux[idx_plus_x], Z9::new(5));

        // Test collision scattering on opposing directions (+X and -X)
        let mut octree_collide = Octree::new(bounds, 10, 4);
        let mut c_left = Couplet::new_baseline(dna);
        let mut c_right = Couplet::new_baseline(dna);
        let idx_minus_x = direction_to_index(-1, 0, 0);

        c_left.flux[idx_plus_x] = Z9::new(3);
        c_right.flux[idx_minus_x] = Z9::new(3);

        octree_collide.insert((-1, 0, 0), c_left);
        octree_collide.insert((1, 0, 0), c_right);
        octree_collide.insert((0, 0, 0), Couplet::new_baseline(dna));

        octree_collide.step(2);

        let c_after = octree_collide.query((0, 0, 0)).unwrap();
        // Opposing +X and -X of equal magnitude (3) streaming from (-1,0,0) and (1,0,0) collide at (0,0,0)
        // and scatter to orthogonal directions (-Y and -Z)
        assert_eq!(c_after.flux[idx_plus_x], Z9::ZERO);
        assert_eq!(c_after.flux[idx_minus_x], Z9::ZERO);
        let (o1, o2) = get_orthogonal_pair(idx_minus_x);
        assert_eq!(c_after.flux[o1], Z9::new(3));
        assert_eq!(c_after.flux[o2], Z9::new(3));
    }

    #[test]
    fn test_aabb_subdivide() {
        let bounds = AABB::new(-16, 16, -16, 16, -16, 16);
        let octants = bounds.subdivide();

        assert_eq!(octants.len(), 8);
        assert_eq!(octants[0], AABB::new(-16, 0, -16, 0, -16, 0));
        assert_eq!(octants[7], AABB::new(1, 16, 1, 16, 1, 16));
    }

    #[test]
    fn test_octree_node_subdivide_and_collapse() {
        let mut pool = vec![OctreeNode::Leaf {
            couplets: Vec::new(),
        }];

        let children = OctreeNode::subdivide_at(&mut pool, 0);
        assert_eq!(children.len(), 8);
        assert_eq!(pool.len(), 9);
        assert!(matches!(pool[0], OctreeNode::Branch { .. }));

        // All children are leaves in uniform resting state (flux = 0)
        let collapsed = OctreeNode::collapse_at(&mut pool, 0);
        assert!(collapsed);
        assert!(matches!(pool[0], OctreeNode::Leaf { .. }));
    }

    #[test]
    fn test_octree_node_collapse_fails_when_active() {
        let mut pool = vec![OctreeNode::Leaf {
            couplets: Vec::new(),
        }];

        let children = OctreeNode::subdivide_at(&mut pool, 0);

        // Inject active couplet (flux != 0) into child 0
        let dna = Tpes::new(2, 2, 1, 1);
        let mut active_couplet = Couplet::new_baseline(dna);
        active_couplet.flux[22] = Z9::new(5);

        if let OctreeNode::Leaf { ref mut couplets } = pool[children[0]] {
            couplets.push(((1, 1, 1), active_couplet));
        }

        let collapsed = OctreeNode::collapse_at(&mut pool, 0);
        assert!(!collapsed);
        assert!(matches!(pool[0], OctreeNode::Branch { .. }));
    }

    #[test]
    fn test_dynamic_step_subdivision() {
        let bounds = AABB::new(-10, 10, -10, 10, -10, 10);
        let mut octree = Octree::new(bounds, 10, 4);

        let dna = Tpes::new(2, 2, 1, 1);
        let mut c1 = Couplet::new_baseline(dna);
        let c2 = Couplet::new_baseline(dna);

        c1.pressure = Z9::new(8);

        octree.insert((1, 1, 1), c1);
        octree.insert((5, 5, 5), c2);

        // Step with flux threshold = 2
        octree.step(2);

        let t = octree.telemetry();
        assert_eq!(t.total_couplets, 2);
        assert!(t.max_depth > 0);
    }

    #[test]
    fn test_monopole_drain_and_unbinding() {
        use crate::stamp::stamp_electron;

        let bounds = AABB::new(-16, 16, -16, 16, -16, 16);
        let mut octree = Octree::new(bounds, 4, 8);

        // Stamp Electron (ID 1) at (0, 0, 0)
        stamp_electron(&mut octree, (0, 0, 0));

        // Insert ID 2 Space couplet adjacent at (1, 0, 0)
        let element_zero_dna = Tpes::new(2, 2, 1, 1);
        octree.insert((1, 0, 0), Couplet::new_baseline(element_zero_dna));

        assert_eq!(octree.telemetry().total_couplets, 2);

        // Step simulation: Monopole Drain unbinds ID 2 at (1, 0, 0)
        octree.step(2);

        // ID 2 node at (1, 0, 0) must be deleted from lattice
        assert!(octree.query((1, 0, 0)).is_none());

        // ID 1 Electron at (0, 0, 0) absorbed energy and ejected flux tangentially
        let id1 = octree.query((0, 0, 0)).expect("Electron at (0,0,0) should exist");
        assert!(id1.energy > Z9::new(1)); // Energy increased from initial 1
        let equatorial_flux = id1.flux[22] + id1.flux[4] + id1.flux[16] + id1.flux[10];
        assert!(equatorial_flux > Z9::ZERO); // Ejected tangentially along equator

        // Toroidal boundary recycling injected a new ID 2 node on outer boundary at (16, 0, 0)
        assert!(octree.query((16, 0, 0)).is_some());
        assert_eq!(octree.telemetry().total_couplets, 2); // Constant total volume
    }

    #[test]
    fn test_geometric_gravity_subdivision() {
        use crate::stamp::stamp_electron;

        let bounds = AABB::new(-16, 16, -16, 16, -16, 16);
        let mut octree = Octree::new(bounds, 4, 8);

        stamp_electron(&mut octree, (0, 0, 0));

        let element_zero_dna = Tpes::new(2, 2, 1, 1);
        // Insert ID 2 node at dist = 2 (close to drain)
        octree.insert((2, 0, 0), Couplet::new_baseline(element_zero_dna));

        octree.step(2);

        let t = octree.telemetry();
        // Close to drain (dist = 2), target_depth = 9 - 2 = 7
        assert!(t.max_depth >= 7);
    }
}
