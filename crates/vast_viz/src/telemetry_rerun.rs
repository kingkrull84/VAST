use vast_core::lattice::octree::Octree;
use rerun::blueprint::{Blueprint, BlueprintActivation, Spatial3DView};
use rerun::{Boxes3D, Color, FillMode, Points3D, RecordingStream, RecordingStreamBuilder};

pub struct TelemetryRerun {
    rec: RecordingStream,
}

impl TelemetryRerun {
    pub fn new(app_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let rec = RecordingStreamBuilder::new(app_name).spawn()?;

        let view = Spatial3DView::new("3D Space View")
            .with_origin("/")
            .with_contents(["/**"]);
        let blueprint = Blueprint::new(view);
        blueprint.send(&rec, BlueprintActivation::default())?;

        // Set default bounds to (-16, 16) centered at origin (0,0,0)
        let default_bounds = Boxes3D::from_centers_and_half_sizes([(0.0, 0.0, 0.0)], [(16.0, 16.0, 16.0)])
            .with_fill_mode(FillMode::MajorWireframe);
        rec.log_static("world/bounds", &default_bounds)?;

        Ok(Self { rec })
    }

    pub fn log_step(&self, cycle: i64, octree: &Octree) -> Result<(), Box<dyn std::error::Error>> {
        self.rec.set_time_sequence("cycle", cycle);

        let active_leaves = octree.collect_active_leaves();

        let mut positions = Vec::new();
        let mut colors = Vec::new();
        let mut labels = Vec::new();
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
                let flux_val = couplet.total_flux().value();
                colors.push(flux_to_color(flux_val));
                labels.push(format!("{}", flux_val));
            }
        }

        if !positions.is_empty() {
            let points = Points3D::new(positions)
                .with_colors(colors)
                .with_radii([0.5])
                .with_labels(labels);
            self.rec.log("world/couplets", &points)?;
        }

        if !box_centers.is_empty() {
            let boxes = Boxes3D::from_centers_and_half_sizes(box_centers, box_half_sizes)
                .with_fill_mode(FillMode::MajorWireframe);
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
