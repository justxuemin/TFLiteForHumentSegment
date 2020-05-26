#version 330

vec2 aVertexPosition;
vec2 aPlotPosition;
vec2 vPosition;
uniform float uScaleTexture;
const vec2 cMapViewToTexture = vec2(0.5, 0.5);

void main(void) {
    gl_Position = vec4(aVertexPosition, 1.0, 1.0);
    vPosition = vec2(uScaleTexture, uScaleTexture) * (aVertexPosition * cMapViewToTexture + cMapViewToTexture);
}