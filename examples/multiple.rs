use bevy::prelude::*;
use bevy_fsc_point_cloud::{OpdLoader, PointCloudAsset, PotreePointCloud};

// usage: cargo run --example multiple -- <count> <frames> <case>
// example: cargo run --example multiple -- 2 150 15

// bash automated run of all cases with collected results at the end:
// for i in {0..23}; do cargo run --example multiple -- 2 150 $i; results[$i]=$?; done; for i in {0..23}; do echo "$i  ${results[$i]}"; done;

// case | direct |  auto | early | delay |  same | crash
//    0 |  false | false | false | false | false | false
//    1 |  false | false | false | false |  true | false
//    2 |  false | false | false |  true | false | false
//    3 |  false | false | false |  true |  true | false
//    4 |  false | false |  true | false | false | false
//    5 |  false | false |  true | false |  true | false
//    6 |  false | false |  true |  true | false |  true
//    7 |  false | false |  true |  true |  true |  true
//    8 |  false |  true | false | false | false | false
//    9 |  false |  true | false | false |  true | false
//   10 |  false |  true | false |  true | false | false
//   11 |  false |  true | false |  true |  true |  true
//   12 |  false |  true |  true | false | false | false
//   13 |  false |  true |  true | false |  true | false
//   14 |  false |  true |  true |  true | false |  true
//   15 |  false |  true |  true |  true |  true |  true
//   16 |   true | false | false | false | false | false
//   17 |   true | false | false | false |  true | false
//   18 |   true | false | false |  true | false |  true
//   19 |   true | false | false |  true |  true |  true
//   20 |   true | false |  true | false | false | false
//   21 |   true | false |  true | false |  true | false
//   22 |   true | false |  true |  true | false |  true
//   23 |   true | false |  true |  true |  true |  true
// direct && auto doesnt make sense

// crash = delay && (direct || early || (auto && same))

// Summary: it can only ever crash when spawning point clouds on separate frames.
// Furthermore, the only cases when it crashes are when skipping the AssetServer,
// or when pre-loading the asset and holding a Handle for later,
// or when using the AssetServer but loading the same path twice.

#[derive(Clone)]
struct LoadConfig {
    /// if `direct`, we skip the `AssetServer` and send the byte blob directly into Assets<PointCloudAsset>
    /// else, we give to `AssetServer` and let it track it
    direct: bool,
    /// if `auto`, we give `AssetServer` the path and let it read it
    /// else, we read the bytes from the files ourselves
    auto: bool,
    /// if `early`, then we load the asset at frame zero and hold a handle until its spawned at frame `delay`
    /// else, load the asset at frame `delay` right before spawning the point cloud
    early: bool,
    /// which frame to spawn the point cloud entity on
    delay: u32,
    /// the name of the point cloud opd file in the assets directory
    name: String,
}

#[derive(Resource, Clone)]
struct LoadConfigs {
    configs: Vec<LoadConfig>,
}

fn main() {
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect();
    let &[count, frames, case] = args.as_slice() else {
        println!("Usage: cargo run --example multiple -- <count> <frames> <case>");
        println!("Example: cargo run --example multiple -- 2 150 15");
        return;
    };
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin::default()),
            bevy_fsc_point_cloud::PointCloudPlugin,
        ))
        .insert_resource(load_configs(count, frames, case))
        .add_systems(Startup, startup)
        .add_systems(Update, spawn_replays)
        .run();
}

fn load_configs(count: u32, frames: u32, case: u32) -> LoadConfigs {
    let direct = (case & 0b10000) != 0;
    let auto = (case & 0b1000) != 0;
    let early = (case & 0b100) != 0;
    let delay = (case & 0b10) != 0;
    let same = (case & 0b1) != 0;
    println!("case: {case}\ncount: {count}\nframes: {frames}\ndirect: {direct}\nauto: {auto}\nearly: {early}\ndelay: {delay}\nsame: {same}");

    LoadConfigs {
        configs: (0..count)
            .map(|i| {
                (
                    frames * (delay as u32 * i + 1),
                    format!("replay{}.opd", !same as u32 * i),
                )
            })
            .map(|(delay, name)| LoadConfig {
                direct,
                auto,
                early,
                delay,
                name,
            })
            .collect(),
    }
}

fn load(
    asset_server: &AssetServer,
    assets: &mut Assets<PointCloudAsset>,
    config: &LoadConfig,
) -> Handle<PointCloudAsset> {
    if config.auto {
        assert!(!config.direct);
        asset_server.load(&config.name)
    } else {
        let path = std::env::current_dir()
            .unwrap()
            .join("assets")
            .join(&config.name);
        let mut point_cloud_bytes = Vec::new();
        std::io::Read::read_to_end(
            &mut std::fs::File::open(&path).unwrap(),
            &mut point_cloud_bytes,
        )
        .unwrap();
        let point_cloud =
            futures_lite::future::block_on(OpdLoader::load_opd(point_cloud_bytes.as_slice()))
                .unwrap();
        if config.direct {
            assets.add(point_cloud)
        } else {
            asset_server.add(point_cloud)
        }
    }
}

fn startup(mut commands: Commands) {
    commands.spawn(Camera3dBundle::default()).insert(
        Transform::from_translation(Vec3::new(0.0, 20.0, 80.0)).looking_at(Vec3::ZERO, Vec3::Y),
    );
}

fn spawn_replays(
    mut commands: Commands,
    frames: Res<bevy::core::FrameCount>,
    mut handles: Local<Vec<Handle<PointCloudAsset>>>,
    mut assets: ResMut<Assets<PointCloudAsset>>,
    mut app_exit_events: ResMut<Events<bevy::app::AppExit>>,
    configs: Res<LoadConfigs>,
    asset_server: Res<AssetServer>,
) {
    let configs = &configs.configs;
    if frames.0 == 1 {
        *handles = configs
            .iter()
            .map(|config| {
                if config.early {
                    load(&asset_server, &mut assets, config)
                } else {
                    Handle::default()
                }
            })
            .collect();
    }
    for (i, config) in configs.iter().enumerate() {
        if config.delay == frames.0 {
            let point_cloud = if config.early {
                handles[i].clone()
            } else {
                load(&asset_server, &mut assets, config)
            };
            commands
                .spawn(PotreePointCloud {
                    mesh: point_cloud,
                    point_size: 2.0,
                })
                .insert(SpatialBundle {
                    transform: Transform::from_rotation(Quat::from_rotation_x(
                        -std::f32::consts::FRAC_PI_2,
                    ))
                    .with_translation(Vec3::new(0.0, 0.0, -100.0 * i as f32)),
                    ..Default::default()
                });
        }
    }
    if frames.0 == configs.first().unwrap().delay * 3 + configs.last().unwrap().delay {
        app_exit_events.send(bevy::app::AppExit);
    }
}
