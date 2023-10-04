#include "BlockA.h"

namespace BlockA
{
	time_point<high_resolution_clock> Time::Now()
	{
		return high_resolution_clock::now();
	}

	double Time::DeltaNano(const time_point<high_resolution_clock>& t)
	{
		return double(duration_cast<nanoseconds>(high_resolution_clock::now() - t).count());
	}

	double Time::DeltaNano(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1)
	{
		return double(duration_cast<nanoseconds>(t1 - t0).count());
	}

	double Time::DeltaMicro(const time_point<high_resolution_clock>& t)
	{
		return double(duration_cast<nanoseconds>(high_resolution_clock::now() - t).count()) / 1000.0;
	}

	double Time::DeltaMicro(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1)
	{
		return double(duration_cast<nanoseconds>(t1 - t0).count()) / 1000.0;
	}

	double Time::DeltaMili(const time_point<high_resolution_clock>& t)
	{
		return double(duration_cast<nanoseconds>(high_resolution_clock::now() - t).count()) / 1000000.0;
	}

	double Time::DeltaMili(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1)
	{
		return double(duration_cast<nanoseconds>(t1 - t0).count()) / 1000000.0;
	}

	Time::Time(const string& name)
		: name(name)
	{
		startedTime = Now();
		touchedTime = startedTime;
	}

	void Time::Stop()
	{
		if (name.empty() == false)
		{
			cout << "[" << name << "] ";
		}
		cout << DeltaMili(startedTime) << " miliseconds" << endl;
	}

	void Time::Touch()
	{
		touchCount++;
		auto now = Now();

		if (name.empty() == false)
		{
			cout << "[" << name << " : " << touchCount << "] ";
		}
		else
		{
			cout << "[" << touchCount << "] ";
		}
		cout << DeltaMili(touchedTime, now) << " miliseconds" << endl;

		touchedTime = now;
	}

    Shader::Shader(const char* vertexPath, const char* fragmentPath) {
        // Load the vertex shader from file
        std::ifstream vertexFile(vertexPath);
        if (!vertexFile) {
            std::cout << "Failed to open vertex shader file" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::stringstream vertexStream;
        vertexStream << vertexFile.rdbuf();
        std::string vertexSource = vertexStream.str();
        const char* vertexSourcePtr = vertexSource.c_str();

        // Load the fragment shader from file
        std::ifstream fragmentFile(fragmentPath);
        if (!fragmentFile) {
            std::cout << "Failed to open fragment shader file" << std::endl;
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
        m_ID = glCreateProgram();
        glAttachShader(m_ID, vertexShader);
        glAttachShader(m_ID, fragmentShader);
        glLinkProgram(m_ID);
        checkCompileErrors(m_ID, "PROGRAM");

        // Delete the shaders
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }

    Shader::~Shader() {
        glDeleteProgram(m_ID);
    }

    void Shader::use() {
        glUseProgram(m_ID);
    }

    void Shader::setInt(const char* name, int value) {
        glUniform1i(glGetUniformLocation(m_ID, name), value);
    }

    void Shader::setFloat(const char* name, float value) {
        glUniform1f(glGetUniformLocation(m_ID, name), value);
    }

    void Shader::checkCompileErrors(unsigned int shader, const char* type) {
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

	Application::Application(int width, int height, const string& windowTitle)
	{
		window = new Window(width, height, windowTitle);
	}

	Application::~Application()
	{
		SAFE_DELETE(window)
	}

	void Application::OnInitialize(function<void()> onInitialize)
	{
		onInitializeFunction = onInitialize;
	}

	void Application::OnUpdate(function<void(float)> onUpdate)
	{
		onUpdateFunction = onUpdate;
	}

	void Application::OnTerminate(function<void()> onTerminate)
	{
		onTerminateFunction = onTerminate;
	}

	void Application::Run()
	{
        // Our state
        bool show_demo_window = true;
        bool show_another_window = false;
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);





		bool appFinisihed = false;

		if (onInitializeFunction != nullptr)
		{
			onInitializeFunction();
		}

		Time time("main");
		auto now = time.Now();
		auto lastTime = now;

		while (appFinisihed == false)
		{
			auto now = time.Now();
			auto timeDelta = time.DeltaMili(lastTime, now);

			appFinisihed = true;
			if (window->ShouldClose() == false)
			{
				appFinisihed = false;

				//window->MakeCurrent();
				window->ProcessEvents();

				if (onUpdateFunction != nullptr)
				{
					onUpdateFunction((float)timeDelta);
				}

                // Start the Dear ImGui frame
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
                if (show_demo_window)
                    ImGui::ShowDemoWindow(&show_demo_window);

                // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
                {
                    static float f = 0.0f;
                    static int counter = 0;

                    ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

                    ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
                    ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
                    ImGui::Checkbox("Another Window", &show_another_window);

                    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
                    ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

                    if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                        counter++;
                    ImGui::SameLine();
                    ImGui::Text("counter = %d", counter);

                    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / window->GetIO()->Framerate, window->GetIO()->Framerate);
                    ImGui::End();
                }

                // 3. Show another simple window.
                if (show_another_window)
                {
                    ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
                    ImGui::Text("Hello from another window!");
                    if (ImGui::Button("Close Me"))
                        show_another_window = false;
                    ImGui::End();
                }

                // Rendering
                ImGui::Render();
                int display_w, display_h;
                glfwGetFramebufferSize(window->GetGLFWWindow(), &display_w, &display_h);
                glViewport(0, 0, display_w, display_h);
                //glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
                //glClear(GL_COLOR_BUFFER_BIT);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			}

			window->SwapBuffers();

			lastTime = now;
		}

		if (onTerminateFunction != nullptr)
		{
			onTerminateFunction();
		}
	}
}
