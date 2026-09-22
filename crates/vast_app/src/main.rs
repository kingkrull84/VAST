use vast_core::lattice::couplet::Couplet;
use vast_core::lattice::octree::{Octree, AABB};
use vast_core::stamp::stamp_electron;
use vast_core::tpes::Tpes;
use vast_viz::TelemetryRerun;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VAST Engine Simulation Starting ===");

    // FIX: Safely offload Rerun's blocking initialization to a dedicated OS thread
    // We map the error to a String so it safely crosses the thread boundary
    let rerun_telemetry = match tokio::task::spawn_blocking(|| {
        TelemetryRerun::new("vast_app").map_err(|e| e.to_string())
    }).await {
        Ok(Ok(t)) => {
            println!("Rerun web viewer listening on http://0.0.0.0:9090");
            Some(t)
        }
        Ok(Err(e)) => {
            eprintln!("Warning: Failed to start Rerun web server: {}", e);
            None
        }
        Err(e) => {
            eprintln!("Warning: Tokio spawn_blocking failed: {}", e);
            None
        }
    };

    // Define 3D bounding box for simulation space (-16..16 on all axes)
    let bounds = AABB {
        min_x: -16,
        max_x: 16,
        min_y: -16,
        max_y: 16,
        min_z: -16,
        max_z: 16,
    };
    println!("Initialized Sparse Octree lattice bounds: {:?}", bounds);

    let capacity = 64_000; 
    let max_depth = 8;
    let mut octree = Octree::new(bounds, capacity, max_depth);

    // Fill ocean bounds with resting ID 2 couplets (Element Zero)
    for x in -16..=16 {
        for y in -16..=16 {
            for z in -16..=16 {
                let elem_zero = Couplet::new(2, Tpes::new(2, 2, 1, 1));
                octree.insert((x, y, z), elem_zero);
            }
        }
    }
    println!("Filled [-16, 16] AABB bounds with resting ID 2 couplets (Element Zero)");

    // Stamp Electron at origin
    stamp_electron(&mut octree, (0, 0, 0));
    println!("Stamped Electron (1:0:1:1) at origin (0, 0, 0)");

    let total_cycles = 10;
    for cycle in 1..=total_cycles {
        if let Some(ref t) = rerun_telemetry {
            if let Err(e) = t.log_step(cycle, &octree) {
                eprintln!("Error logging cycle {} Rerun telemetry: {}", cycle, e);
            }
        }
    }

    println!("=== VAST Engine Simulation Completed Successfully ===");
    println!("\n3D Viewport is active on port 9090. Press Ctrl+C in terminal to exit.");

    tokio::signal::ctrl_c().await?;
    Ok(())
}