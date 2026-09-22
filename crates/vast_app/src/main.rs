use vast_core::lattice::couplet::Couplet;
use vast_core::lattice::octree::{Octree, AABB};
use vast_core::stamp::stamp_electron;
use vast_core::tpes::Tpes;
use vast_viz::TelemetryRerun;

#[tokio::main]
async fn main() {
    println!("=== VAST Engine Simulation Starting ===");

    // Define 3D bounding box for simulation space (-16..16 on all axes)
    let bounds = AABB::new(-16, 16, -16, 16, -16, 16);
    let mut octree = Octree::new(bounds, 4, 8);

    println!("Initialized Sparse Octree lattice bounds: {:?}", bounds);

    // Fill initial [-16, 16] AABB bounds with resting ID 2 couplets (Element Zero)
    let element_zero_dna = Tpes::new(2, 2, 1, 1);
    for x in -16..=16 {
        for y in -16..=16 {
            for z in -16..=16 {
                octree.insert((x, y, z), Couplet::new_baseline(element_zero_dna));
            }
        }
    }
    println!("Filled [-16, 16] AABB bounds with resting ID 2 couplets (Element Zero)");

    // Inject Electron (T:1, P:0, E:1, S:1) into ID 1 vacuum couplet space at origin
    let origin = (0, 0, 0);
    stamp_electron(&mut octree, origin);
    println!("Stamped Electron (1:0:1:1) at origin {:?}", origin);

    // Initialize 3D Visual Telemetry Viewport (Rerun)
    let rerun_telemetry = match TelemetryRerun::new("VAST Simulation Engine") {
        Ok(t) => {
            println!("Initialized Rerun 3D Visual Telemetry stream.");
            Some(t)
        }
        Err(e) => {
            eprintln!("Failed to initialize Rerun telemetry: {}", e);
            None
        }
    };

    // Initial telemetry reading & Rerun log
    let initial_telemetry = octree.telemetry();
    println!("Initial Telemetry: {:?}", initial_telemetry);
    if let Some(ref t) = rerun_telemetry {
        if let Err(e) = t.log_step(0, &octree) {
            eprintln!("Error logging initial Rerun telemetry: {}", e);
        }
    }

    // Simulation cycles stepping through discrete Z/9Z flux updates
    let total_cycles = 5;
    for cycle in 1..=total_cycles {
        octree.step(2);
        let telemetry = octree.telemetry();
        println!("Cycle {:02} Telemetry: {:?}", cycle, telemetry);

        if let Some(ref t) = rerun_telemetry {
            if let Err(e) = t.log_step(cycle, &octree) {
                eprintln!("Error logging cycle {} Rerun telemetry: {}", cycle, e);
            }
        }
    }

    println!("=== VAST Engine Simulation Completed Successfully ===");

    println!("\n3D Viewport is running. Waiting for Ctrl+C to exit...");
    let _ = tokio::signal::ctrl_c().await;
}
