// 单位着色器
#version 330
vec4 vVettex;
vec3 vNormal;
uniform vec4 diffuseColor;
uniform vec3 vLightPosition;
uniform mat4 mvpMatrix;
uniform mat4 mvMatrix;
uniform mat3 normalMatrix;

smooth out vec4 vVaryingColor;
void main(void)
{
    vec3 vEyeNormal = normalMatrix * vNormal;
    vec4 vPosition4 = mvMatrix * vVettex;
    vec3 vPosition3 = vPosition4.xyz / vPosition4.w;
    vec3 vLightDir = normalize(vLightPosition - vPosition3);
    float diff = max(0.0, dot(vEyeNormal, vLightDir));
    vVaryingColor.xyz = diff * diffuseColor.xyz;
    vVaryingColor.a = 1.0;
    gl_Position = mvpMatrix * vVettex;
}