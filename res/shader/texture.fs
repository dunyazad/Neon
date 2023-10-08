#version 330 core

in vec2 texCoord;

uniform sampler2D texture;

out vec4 color;

void main() {
    color = texture2D(texture, texCoord);
}