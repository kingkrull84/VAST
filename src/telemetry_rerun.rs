use crate::lattice::octree::Octree;
use rerun::{Boxes3D, Color, Points3D, RecordingStream, RecordingStreamBuilder};

pub struct TelemetryRerun {
    rec: RecordingStream,
}

impl TelemetryRerun {
    pub fn new(app_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let rec = RecordingStreamBuilder::new(app_name).spawn()?;
        Ok(Self { rec })
    }

    pub fn log_step(&self, cycle: i64, octree: &Octree) -> Result<(), Box<dyn std::error::Error>> {
        self.rec.set_time_sequence("cycle", cycle);

        let active_leaves = octree.collect_active_leaves();

        let mut positions = Vec::new();
        let mut colors = Vec::new();
        let mut box_centers = Vec::new();
        let mut box_half_sizes = Vec::new();

        for (bounds, couplets) in &active_leaves {
            let center_x = (bounds.min_x + bounds.max_x) as f32 / 2.0;
            let center_y = (bounds.min_y + bounds.max_y) as f32 / 2.0;
            let center_z = (bounds.min_z + bounds.max_z) as f32 / 2.0;

            let half_x = (bounds.max_x - bounds.min_x).abs() as f32 / 2.0;
            let half_y = (bounds.max_y - bounds.min_y).abs() as f32 / 2.0;
            let half_z = (bounds.max_z - bounds.min_z).abs() as f32 / 2.0;

            box_centers.push((center_x, center_y, center_z));
            box_half_sizes.push((half_x, half_y, half_z));

            for (pos, couplet) in couplets {
                positions.push((pos.0 as f32, pos.1 as f32, pos.2 as f32));
                let flux_val = couplet.flux.value();
                colors.push(flux_to_color(flux_val));
            }
        }

        if !positions.is_empty() {
            let points = Points3D::new(positions).with_colors(colors);
            self.rec.log("world/couplets", &points)?;
        }

        if !box_centers.is_empty() {
            let boxes = Boxes3D::from_centers_and_half_sizes(box_centers, box_half_sizes);
            self.rec.log("world/leaf_regions", &boxes)?;
        }

        Ok(())
    }
}

fn flux_to_color(flux: u8) -> Color {
    match flux % 9 {
        0 => Color::from_rgb(30, 30, 80),
        1 => Color::from_rgb(0, 100, 255),
        2 => Color::from_rgb(0, 200, 255),
        3 => Color::from_rgb(0, 255, 150),
        4 => Color::from_rgb(100, 255, 0),
        5 => Color::from_rgb(255, 255, 0),
        6 => Color::from_rgb(255, 150, 0),
        7 => Color::from_rgb(255, 50, 0),
        8 => Color::from_rgb(255, 0, 255),
        _ => Color::from_rgb(255, 255, 255),
    }
}
