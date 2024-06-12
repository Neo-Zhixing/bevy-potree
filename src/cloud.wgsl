#import bevy_render::view::View

struct ClippingPlane {
    origin: vec3<f32>,
    unit_normal: vec3<f32>,
    min_sdist: f32,
    max_sdist: f32,
};

struct ClippingPlanes {
    ranges: array<ClippingPlane, 16>,
    num_ranges: u32,
};

struct Model {
    model_transform: mat4x4<f32>,
    point_size_world_space: f32,
};

struct PointOffset {
    position_x: f32,
    position_y: f32,
    position_z: f32,
};

#ifdef ANIMATED
struct AnimationOffset {
    _old_interpolation: f32,
    prev_offsets: array<PointOffset>,
};

struct AnimationOffsetNext {
    interpolation: f32,
    next_offsets: array<PointOffset>,
};

@group(1) @binding(1) var<storage, read> animation_offset: AnimationOffset;
@group(1) @binding(2) var<storage, read> animation_offset_next: AnimationOffsetNext;
#endif

struct Point {
    position_x: f32,
    position_y: f32,
    position_z: f32,
    #ifdef COLORED
    color_r: f32,
    color_g: f32,
    color_b: f32,
    #endif
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> clipping_planes: ClippingPlanes;
@group(2) @binding(0) var<uniform> model: Model;
@group(1) @binding(0) var<storage, read> points: array<Point>;

struct VertexOutput {
    @location(0) out_point_location: vec2<f32>,
    @location(1) out_color: vec3<f32>,
    @builtin(position) vert_position: vec4<f32>,
};

fn discard_vertex() -> VertexOutput {
    var output: VertexOutput;

    let nan = bitcast<f32>(0x7fc00000);
    output.vert_position = vec4<f32>(nan);
    return output;
}

@vertex
fn vertex(
    @location(0) in_position_point: vec2<f32>,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var output: VertexOutput;

    let p = points[instance_index];

    var in_pos = vec3<f32>(p.position_x, p.position_y, p.position_z);
    #ifdef ANIMATED
    var prev_offset: PointOffset = animation_offset.prev_offsets[instance_index];
    var next_offset: PointOffset = animation_offset_next.next_offsets[instance_index];
    
    var prev: vec3<f32> = vec3<f32>(prev_offset.position_x, prev_offset.position_y, prev_offset.position_z);
    var next: vec3<f32> = vec3<f32>(next_offset.position_x, next_offset.position_y, next_offset.position_z);
    
    var interpolated: vec3<f32> = prev + (next - prev) * animation_offset_next.interpolation;
    in_pos += interpolated;
    #endif

    let out_Pos = view.view_proj * model.model_transform * vec4<f32>(in_pos, 1.0);
    if (clipping_planes.num_ranges > 0u) {
        let world_pos4 = model.model_transform * vec4<f32>(in_pos, 1.0);
        let world_pos = world_pos4.xyz / world_pos4.w;

        // Clip any points that falls out of the allowed ranges.
        for (var i = 0u; i < clipping_planes.num_ranges; i++) {
            let range = clipping_planes.ranges[i];
            let sdist_to_plane = dot(world_pos - range.origin, range.unit_normal);
            if (sdist_to_plane < range.min_sdist || sdist_to_plane > range.max_sdist) {
                // DISCARD point
                return discard_vertex();
            }
        }
    }
    #ifdef COLORED
    output.out_color = vec3<f32>(p.color_r, p.color_g, p.color_b);
    #else
    output.out_color = vec3<f32>(p.position_x % 1.0, p.position_y % 1.0, p.position_z % 1.0);
    #endif

    var point_size = vec2<f32>(0.0, 0.0);
    if (view.projection[2][3] == -1.0) {
        let depth = out_Pos.w;
        let one_over_slope = view.projection[1][1];
        point_size = vec2<f32>(0.5 * model.point_size_world_space * one_over_slope);
    } else {
        let a = 2.0 / view.projection[0][0];
        let b = 2.0 / view.projection[1][1];
        let max_scale = max(abs(a), abs(b));
        point_size = vec2<f32>(model.point_size_world_space / max_scale);
    }
    point_size.y *= view.viewport.z / view.viewport.w;

    output.out_point_location = in_position_point;
    output.vert_position = out_Pos + vec4<f32>(in_position_point * point_size, 0.0, 0.0);

    return output;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
    @builtin(frag_depth) frag_depth: f32,
};


@fragment
fn fragment(
    @location(0) in_point_location: vec2<f32>,
    @location(1) in_color: vec3<f32>,
    @builtin(position) frag_coord: vec4<f32>,
) -> FragmentOutput {
    var output: FragmentOutput;

    let uv = in_point_location * 2.0 - 1.0;
    var depth_offset = sqrt(uv.x * uv.x + uv.y * uv.y);
    output.color = vec4<f32>(in_color, 1.0);


    let depth = 1.0 / frag_coord.w; // the world space depth

    if (view.projection[2][3] != -1.0) {
        // orthographic projection
        // projection[2][2] is r = 1.0 / (near - far).
        // This divides the depth offset by (near - far)
        depth_offset *= view.projection[2][2];
    }

    let offseted_depth = depth + model.point_size_world_space * depth_offset;

    let z_near = frag_coord.z * depth;
    let depth_output = z_near / offseted_depth;

    output.depth = depth_output;
    output.frag_depth = depth_output;
    return output;
}
