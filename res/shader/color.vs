#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec4 vColor;

void main() {
    vColor = color;
    gl_Position = projection * view * model * vec4(position.x, position.y, position.z, 1.0);
}