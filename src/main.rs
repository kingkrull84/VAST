pub mod lattice;
pub mod stamp;
pub mod tpes;
pub mod uss;

use lattice::octree::{Octree, AABB};
use stamp::stamp_proton_triad;

fn main() {
    println!("=== VAST Engine Simulation Starting ===");

    // Define 3D bounding box for simulation space (-16..16 on all axes)
    let bounds = AABB::new(-16, 16, -16, 16, -16, 16);
    let mut octree = Octree::new(bounds, 4, 8);

    println!("Initialized Sparse Octree lattice bounds: {:?}", bounds);

    // Inject Proton triad (T:2, P:2, E:1, S:1) into ID 2 couplet space at origin
    let origin = (0, 0, 0);
    stamp_proton_triad(&mut octree, origin);
    println!("Stamped Proton Triad (2:2:1:1) at origin {:?}", origin);

    // Initial telemetry reading
    let initial_telemetry = octree.telemetry();
    println!("Initial Telemetry: {:?}", initial_telemetry);

    // Simulation cycles stepping through discrete Z/9Z flux updates
    let total_cycles = 5;
    for cycle in 1..=total_cycles {
        octree.step();
        let telemetry = octree.telemetry();
        println!("Cycle {:02} Telemetry: {:?}", cycle, telemetry);
    }

    println!("=== VAST Engine Simulation Completed Successfully ===");
}
