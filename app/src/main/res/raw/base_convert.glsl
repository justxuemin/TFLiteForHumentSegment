#version 310 es
#extension GL_OES_EGL_image_external_essl3: enable
precision mediump float;
// need to specify 'mediump' for float
//   layout(local_size_x = 16, local_size_y = 16) in;
layout(local_size_x = 8, local_size_y = 8) in;
//   layout(binding = 0) uniform sampler2D in_data;
layout(binding = 0) uniform samplerExternalOES in_data;
layout(std430) buffer;
layout(binding = 1) buffer Input { float elements[]; } out_data;
void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= img_cx || gid.y >= img_cy ) return;
    vec2 uv = vec2(gl_GlobalInvocationID.xy) /  + img_cx + .0;
    vec4 pixel = texture (in_data, uv);
    int idx = 3 * (gid.y *  img_cx + gid.x);
    // if (gid.x >= 120) pixel.x = 1.0;
    // DEBUG...
    out_data.elements[idx + 0] = pixel.x;
    out_data.elements[idx + 1] = pixel.y;
    out_data.elements[idx + 2] = pixel.z;
}