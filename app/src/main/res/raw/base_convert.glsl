#version 310 es
//#extension GL_OES_EGL_image_external_essl3: enable
precision mediump float;
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0) uniform sampler2D input_texture;
layout(std430) buffer;
layout(binding = 1) buffer Input { float elements[]; } out_data;
void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= 224 || gid.y >= 224)
        return;
    vec4 pixel = texelFetch(input_texture, gid, 0);
    int linear_index = 3 * (gid.y * 224 + gid.x);
    out_data.elements[linear_index + 0] = pixel.x;
    out_data.elements[linear_index + 1] = pixel.y;
    out_data.elements[linear_index + 2] = pixel.z;
}