use crate::PointCloudAsset;
use bevy::{
    asset::{io::Reader, AssetLoader, AsyncReadExt, LoadContext},
    math::Vec3A,
    prelude::*,
    render::{render_asset::RenderAssetUsages, render_resource::PrimitiveTopology},
    utils::{
        thiserror::{self, Error},
        BoxedFuture,
    },
};

#[derive(Default)]
pub struct OpdLoader;

impl OpdLoader {
    pub async fn load_opd(
        bytes: &[u8],
    ) -> Result<PointCloudAsset, nom::Err<nom::error::Error<Vec<u8>>>> {
        let file = opd_parser::parse(bytes).map_err(|e| e.to_owned())?.1;
        let mut positions: Vec<Vec3A> = Vec::new();

        let mut max_position = Vec3A::splat(f32::MIN);
        let mut min_position = Vec3A::splat(f32::MAX);

        for i in file.centroids {
            max_position = max_position.max(i.offset.into());
            min_position = min_position.min(i.offset.into());
            positions.push(i.offset.into());
        }

        let size = max_position - min_position;
        let position_offset: Vec3A = min_position + size / 2.0;
        for position in positions.iter_mut() {
            *position -= position_offset;
        }

        let mut mesh = Mesh::new(PrimitiveTopology::PointList, RenderAssetUsages::all());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

        Ok(PointCloudAsset {
            mesh,
            animation: Some(file.frames),
            animation_scale: file.header.directive.scale.into(),
        })
    }
}

/// Possible errors that can be produced by [`OpdLoader`]
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum OpdLoaderError {
    #[error("Could not load asset: {0}")]
    Io(#[from] std::io::Error),
    #[error("Could not parse Opd: {0}")]
    OpdParseError(#[from] nom::Err<nom::error::Error<Vec<u8>>>),
}

impl AssetLoader for OpdLoader {
    type Asset = PointCloudAsset;
    type Settings = ();
    type Error = OpdLoaderError;

    fn extensions(&self) -> &[&str] {
        &["opd"]
    }

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a (),
        _load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;
            let asset = Self::load_opd(bytes.as_slice()).await?;
            Ok(asset)
        })
    }
}
