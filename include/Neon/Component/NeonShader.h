#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
    class Shader : public ComponentBase {
    public:
        Shader(const string& name, const char* vertexPath, const char* fragmentPath);
        ~Shader();

        void Use();
        void SetUniformInt(const char* name, int value);
        void SetUniformFloat1(const char* name, float value);
		void SetUniformFloat2(const char* name, float values[2]);
		void SetUniformFloat3(const char* name, float values[3]);
		void SetUniformFloat4(const char* name, float values[4]);
        void SetUniformFloat3x3(const char* name, float value[9]);
        void SetUniformFloat4x4(const char* name, float value[16]);

    private:
        GLuint shaderProgram;

        void checkCompileErrors(GLuint shader, const char* type);
    };
}