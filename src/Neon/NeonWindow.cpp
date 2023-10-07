#include <Neon/NeonWindow.h>

namespace Neon
{
    Window::Window(int width, int height, const string& title, Window* sharingWindow)
    {
        // Initialize GLFW
        if (!glfwInit()) {
            std::cout << "Failed to initialize GLFW" << std::endl;
            exit(EXIT_FAILURE);
        }

        // Set GLFW window hints
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        // Create a GLFW window
        if (sharingWindow != nullptr)
        {
            glfwWindow = glfwCreateWindow(width, height, title.c_str(), NULL, sharingWindow->glfwWindow);
        }
        else
        {
            glfwWindow = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);

            if (!glfwWindow) {
                std::cout << "Failed to create GLFW window" << std::endl;
                glfwTerminate();
                exit(EXIT_FAILURE);
            }

            MakeCurrent();

            // Load OpenGL function pointers using GLAD
			if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
                std::cout << "Failed to initialize GLAD" << std::endl;
                glfwTerminate();
                exit(EXIT_FAILURE);
            }
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        io = &ImGui::GetIO(); (void)io;
        io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsLight();

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
        const char* glsl_version = "#version 130";
        ImGui_ImplOpenGL3_Init(glsl_version);
    }

    Window::~Window()
    {
        // Destroy the GLFW window
        glfwDestroyWindow(glfwWindow);

        // Terminate GLFW
        glfwTerminate();
    }

    bool Window::ShouldClose()
    {
        return glfwWindowShouldClose(glfwWindow);
    }

    void Window::MakeCurrent()
    {
        glfwMakeContextCurrent(glfwWindow);
    }

    void Window::ProcessEvents()
    {
        glfwPollEvents();

        // Check if the escape key was pressed
        if (glfwGetKey(glfwWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(glfwWindow, GLFW_TRUE);
        }
    }

    void Window::Update(float timeDelta)
    {
    }

    void Window::SwapBuffers()
    {
        // Swap the front and back buffers
        glfwSwapBuffers(glfwWindow);
    }
}
