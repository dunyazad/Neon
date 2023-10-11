#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec4 vColor;
out vec3 vPosition;
out vec3 vNormal;
out vec3 vViewDirection;

void main() {
    vColor = color;
    gl_Position = projection * view * model * vec4(position, 1.0);
    vec4 p = model * vec4(position, 1.0);
    vPosition = vec3(p) / p.w;
    vNormal = mat3(transpose(inverse(model))) * normal;
    vViewDirection = normalize(-vPosition);
}