#version 330 core

layout(location = 0) in vec3 position;
layout(location = 4) in vec2 texCoordIn;

out vec2 texCoord;

void main() {
    texCoord = texCoordIn;
    gl_Position = vec4(position.x, position.y, position.z, 1.0);
}