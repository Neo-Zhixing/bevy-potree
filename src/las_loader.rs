use bevy::{
    asset::{io::Reader, AssetLoader, AsyncReadExt, LoadContext},
    prelude::*,
    reflect::TypePath,
    render::{
        mesh::MeshVertexAttribute, render_asset::RenderAssetUsages, render_resource::{PrimitiveTopology, VertexFormat}
    },
    utils::thiserror::{self, Error},
};
use las::Read;
use opd_parser::Frames;

pub const ATTRIBUTE_COLOR: MeshVertexAttribute =
    MeshVertexAttribute::new("Vertex_Color", 1, VertexFormat::Float32x3);

#[repr(transparent)]
struct Point {
    inner: [f32; 3],
}
impl From<[f32; 3]> for Point {
    fn from(inner: [f32; 3]) -> Self {
        Self { inner }
    }
}
impl Point {
    pub fn min(&self, other: &Self) -> Self {
        Point {
            inner: [
                self.inner[0].min(other.inner[0]),
                self.inner[1].min(other.inner[1]),
                self.inner[2].min(other.inner[2]),
            ],
        }
    }
    pub fn max(&self, other: &Self) -> Self {
        Point {
            inner: [
                self.inner[0].max(other.inner[0]),
                self.inner[1].max(other.inner[1]),
                self.inner[2].max(other.inner[2]),
            ],
        }
    }
}

#[derive(Asset, Clone, TypePath)]
pub struct PointCloudAsset {
    pub mesh: Mesh,
    pub animation: Option<Frames>,
    pub animation_scale: Vec3,
}

impl PointCloudAsset {
    pub fn animation_duration(&self) -> Option<f32> {
        match &self.animation {
            Some(Frames::I8(frames)) => Some(frames.last().unwrap().time / 1000.),
            _ => None,
        }
    }
}

/// Possible errors that can be produced by [`LasLoader`]
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum LasLoaderError {
    #[error("Could not load asset: {0}")]
    Io(#[from] std::io::Error),
    #[error("Could not parse Las: {0}")]
    RonSpannedError(#[from] las::Error),
}

#[derive(Default)]
pub struct LasLoader;
impl AssetLoader for LasLoader {
    type Asset = PointCloudAsset;
    type Settings = ();
    type Error = LasLoaderError;

    fn extensions(&self) -> &[&str] {
        &["las", "laz"]
    }

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a (),
        _load_context: &'a mut LoadContext,
    ) -> bevy::utils::BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;
            let mut reader = las::Reader::new(std::io::Cursor::new(bytes))?;
            let mut mesh = Mesh::new(PrimitiveTopology::PointList, RenderAssetUsages::all());
            let mut max: Point = [f32::MIN; 3].into();
            let mut min: Point = [f32::MAX; 3].into();
            let (mut positions, colors): (Vec<_>, Vec<_>) = reader
                .points()
                .map(|a| {
                    let p = a.unwrap();
                    let position = {
                        let p: Point = [p.x as f32, p.z as f32, p.y as f32].into();
                        min = min.min(&p);
                        max = max.max(&p);
                        p.inner
                    };
                    let color = {
                        if let Some(color) = &p.color {
                            Vec3::new(
                                color.red as f32 / u16::MAX as f32,
                                color.green as f32 / u16::MAX as f32,
                                color.blue as f32 / u16::MAX as f32,
                            )
                        } else {
                            let intensity = p.intensity as f32 * 0.01;
                            Vec3::new(intensity, intensity, intensity)
                        }
                    };
                    (position, color)
                })
                .unzip();
            let aabb = [
                max.inner[0] - min.inner[0],
                max.inner[1] - min.inner[1],
                max.inner[2] - min.inner[2],
            ];

            // Normalize the positions
            let scale = aabb[0].max(aabb[1]).max(aabb[2]);
            for i in positions.iter_mut() {
                i[0] -= min.inner[0];
                i[1] -= min.inner[1];
                i[2] -= min.inner[2];
                i[0] /= scale;
                i[1] /= scale;
                i[2] /= scale;
            }
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_attribute(ATTRIBUTE_COLOR, colors);
            let asset = PointCloudAsset {
                mesh,
                animation: None,
                animation_scale: Vec3::default(),
            };
            Ok(asset)
        })
    }
}
