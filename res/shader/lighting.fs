#version 330 core

uniform vec3 cameraPosition;
uniform vec3 lightPosition;
uniform vec3 lightDirection;
uniform vec3 lightColor;

//uniform float ambientStrength;
//uniform float specularStrength;
//uniform float shininess;
uniform float ambientStrength = 0.1; // Strength of the ambient light
uniform float specularStrength = 1.0; // Strength of the specular reflection
uniform float shininess = 32.0; // Shininess factor, controls the size of the specular highlights

in vec3 FragPos;
in vec3 Normal;
in vec4 vColor;

out vec4 FragColor;   // Output color

void main() {
    //// Calculate ambient lighting
    //float ambientStrength = 0.1;
    //vec3 ambient = ambientStrength * lightColor;

    //// Calculate diffuse lighting
    //vec3 lightDir = normalize(lightPosition - fragPosition);
    //float diff = max(dot(vNormal, lightDir), 0.0);
    //vec3 diffuse = diff * lightColor;

    //// Calculate specular lighting
    //float specularStrength = 0.5;
    //vec3 viewDir = normalize(-fragPosition);
    //vec3 reflectDir = reflect(-lightDir, vNormal);
    //float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32); // Shininess factor
    //vec3 specular = specularStrength * spec * lightColor;

    //// Combine ambient, diffuse, and specular lighting
    //vec3 result = (ambient + diffuse + specular) * vec3(vColor);

    //FragColor = vec4(result, 1.0);




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
