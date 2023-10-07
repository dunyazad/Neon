#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
    class Shader {
    public:
        Shader(const char* vertexPath, const char* fragmentPath);
        ~Shader();

        void use();
        void setInt(const char* name, int value);
        void setFloat1(const char* name, float value);
		void setFloat2(const char* name, float values[2]);
		void setFloat3(const char* name, float values[3]);
		void setFloat4(const char* name, float values[4]);

    private:
        GLuint shaderProgram;

        void checkCompileErrors(GLuint shader, const char* type);
    };
}