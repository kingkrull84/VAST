use rerun::{RecordingStreamBuilder, Points3D, Color};
use vast_core::lattice::octree::Octree;

pub struct TelemetryRerun {
    rec: rerun::RecordingStream,
}

impl TelemetryRerun {
    pub fn new(app_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let rec = RecordingStreamBuilder::new(app_name).serve(
            "0.0.0.0",
            Default::default(),
            Default::default(),
            rerun::MemoryLimit::UNLIMITED,
            false,
        )?;

        Ok(Self { rec })
    }

    pub fn log_step(
        &self,
        cycle: usize,
        _octree: &Octree,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Set the current simulation cycle
        self.rec.set_time_sequence("step", cycle as i64);

        // Send a cyan 3D point to represent the electron at the origin
        self.rec.log(
            "vast_simulation/electron",
            &Points3D::new([(0.0, 0.0, 0.0)])
                .with_colors([Color::from_rgb(0, 255, 255)])
                .with_radii([0.5]),
        )?;

        Ok(())
    }
}