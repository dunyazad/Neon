#include <Neon/Component/NeonShader.h>

namespace Neon
{
    Shader::Shader(const string& name, const char* vertexPath, const char* fragmentPath)
        : ComponentBase(name)
    {
        // Load the vertex shader from file
        std::ifstream vertexFile(vertexPath);
        if (!vertexFile) {
            std::cout << "Failed to open vertex shader file : " << vertexPath << std::endl;
            exit(EXIT_FAILURE);
        }
        std::stringstream vertexStream;
        vertexStream << vertexFile.rdbuf();
        std::string vertexSource = vertexStream.str();
        const char* vertexSourcePtr = vertexSource.c_str();

        // Load the fragment shader from file
        std::ifstream fragmentFile(fragmentPath);
        if (!fragmentFile) {
            std::cout << "Failed to open fragment shader file : " << fragmentPath << std::endl;
            exit(EXIT_FAILURE);
        }
        std::stringstream fragmentStream;
        fragmentStream << fragmentFile.rdbuf();
        std::string fragmentSource = fragmentStream.str();
        const char* fragmentSourcePtr = fragmentSource.c_str();

        // Compile the shaders
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexSourcePtr, NULL);
        glCompileShader(vertexShader);
        checkCompileErrors(vertexShader, "VERTEX");

        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentSourcePtr, NULL);
        glCompileShader(fragmentShader);
        checkCompileErrors(fragmentShader, "FRAGMENT");

        // Link the shader program
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        checkCompileErrors(shaderProgram, "PROGRAM");

        // Delete the shaders
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }

    Shader::~Shader() {
        glDeleteProgram(shaderProgram);
    }

    void Shader::Use() {
        glUseProgram(shaderProgram);
    }

    void Shader::SetUniformInt(const char* name, int value) {
        auto location = glGetUniformLocation(shaderProgram, name);
        if (-1 != location)
        {
            glUniform1i(location, value);
        }
    }

    void Shader::SetUniformFloat1(const char* name, float value) {
        auto location = glGetUniformLocation(shaderProgram, name);
        if (-1 != location)
        {
            glUniform1f(location, value);
        }
    }

    void Shader::SetUniformFloat2(const char* name, float values[2]) {
        auto location = glGetUniformLocation(shaderProgram, name);
        if (-1 != location)
        {
            glUniform2f(location, values[0], values[1]);
        }
    }

    void Shader::SetUniformFloat3(const char* name, float values[3]) {
        auto location = glGetUniformLocation(shaderProgram, name);
        if (-1 != location)
        {
            glUniform3f(location, values[0], values[1], values[2]);
        }
    }

    void Shader::SetUniformFloat4(const char* name, float values[4]) {
        auto location = glGetUniformLocation(shaderProgram, name);
        if (-1 != location)
        {
            glUniform4f(location, values[0], values[1], values[2], values[3]);
        }
    }

    void Shader::SetUniformFloat3x3(const char* name, float value[9]) {
        auto location = glGetUniformLocation(shaderProgram, name);
        if (-1 != location)
        {
            glUniformMatrix3fv(location, 1, false, (const float*)value);
        }
    }
    
    void Shader::SetUniformFloat4x4(const char* name, float value[16]) {
        auto location = glGetUniformLocation(shaderProgram, name);
        if (-1 != location)
        {
            glUniformMatrix4fv(location, 1, false, (const float*)value);

            //CheckGLError();
        }
    }

    void Shader::checkCompileErrors(GLuint shader, const char* type) {
        int success;
        char infoLog[1024];
        if (strcmp(type, "PROGRAM") != 0) {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "Failed to compile shader of type " << type << std::endl;
                std::cout << infoLog << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "Failed to link shader program" << std::endl;
                std::cout << infoLog << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}
