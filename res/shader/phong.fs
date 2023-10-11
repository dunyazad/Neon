#version 330 core

uniform vec3 lightPos;
uniform vec3 lightDirection;
uniform vec3 lightColor;

//uniform float ambientStrength;
//uniform float specularStrength;
//uniform float shininess;
uniform float ambientStrength = 0.1; // Strength of the ambient light
uniform float specularStrength = 1.0; // Strength of the specular reflection
uniform float shininess = 32.0; // Shininess factor, controls the size of the specular highlights

in vec4 vColor;
in vec3 vPosition;
in vec3 vNormal;
in vec3 vViewDirection;

out vec4 color;

void main() {
    vec3 lightDir = normalize(lightPos - vPosition);
    vec3 ambient = ambientStrength * lightColor;
    float diff = max(dot(vNormal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 reflectDir = reflect(-lightDir, vNormal);
    float spec = pow(max(dot(vViewDirection, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * vec3(vColor);

    color = vec4(result, 1.0);
}
