#version 330 core

uniform int fillMode = 0;

uniform vec3 cameraPosition;
uniform vec3 lightPosition;
uniform vec3 lightDirection;
uniform vec3 lightColor;

uniform float ambientStrength = 0.1;
uniform float specularStrength = 1.0;
uniform float shininess = 32.0;

in vec3 FragPos;
in vec3 Normal;
in vec4 vColor;

out vec4 FragColor;

void main() {
    if(fillMode == 0)
    {
        vec3 ambient = ambientStrength * lightColor;

        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPosition - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;

        vec3 viewDir = normalize(cameraPosition - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;

        vec3 result = (ambient + diffuse + specular) * vColor.rgb;
        FragColor = vec4(result, vColor.a);
    }
    else if (fillMode != 0)
    {
        FragColor = vColor;
    }
}
