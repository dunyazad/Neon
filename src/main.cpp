#include <iostream>

#include "BlockA.h"

int main()
{
    BlockA::Window* window = nullptr;
    BlockA::Shader* shader = nullptr;
    
    unsigned int VBO, VAO;

    BlockA::Application app;
    app.SetResourceRoot("C:/Workspace/Blocks/BlockA/res");

    app.OnInitialize([&]() {
        {
            // Define the vertices of the triangle
            float vertices[] = {
                 0.0f,  0.5f, 0.0f,
                 0.5f, -0.5f, 0.0f,
                -0.5f, -0.5f, 0.0f
            };

            // Create the vertex buffer object and vertex array object
            glGenBuffers(1, &VBO);
            glGenVertexArrays(1, &VAO);

            // Bind the vertex array object and vertex buffer object
            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);

            // Copy the vertices data into the vertex buffer object
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

            // Specify the vertex attribute pointers
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);

            //// Create the shader program
            shader = new BlockA::Shader((app.GetResourceRoot() + "/shader/vertexShader.glsl").c_str(), (app.GetResourceRoot() + "/shader/fragmentShader.glsl").c_str());
        }
        });





    app.OnUpdate([&](float timeDelta) {
        if (window == window)
        {
            // Clear the screen
            glClearColor(0.3f, 0.5f, 0.7f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            //// Draw the triangle
            shader->use();
            glBindVertexArray(VAO);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        });






    app.OnTerminate([&]() {
        SAFE_DELETE(shader);

        // Cleanup
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        });

    app.Run();

    return 0;
}
