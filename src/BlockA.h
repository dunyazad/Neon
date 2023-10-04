#pragma once

#include "BlockA_Common.h"

using namespace std::chrono;

namespace BlockA
{
	class Time
	{
	public:
		static time_point<high_resolution_clock> Now();
		static double DeltaNano(const time_point<high_resolution_clock>& t);
		static double DeltaNano(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1);

		static double DeltaMicro(const time_point<high_resolution_clock>& t);
		static double DeltaMicro(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1);

		static double DeltaMili(const time_point<high_resolution_clock>& t);
		static double DeltaMili(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1);


		Time(const string& name);
		void Touch();
		void Stop();

	protected:
		string name;
		time_point<high_resolution_clock> startedTime;
		int touchCount = 0;
		time_point<high_resolution_clock> touchedTime;
	};

    class Shader {
    public:
        Shader(const char* vertexPath, const char* fragmentPath);
        ~Shader();

        void use();
        void setInt(const char* name, int value);
        void setFloat(const char* name, float value);

    private:
        unsigned int m_ID;

        void checkCompileErrors(unsigned int shader, const char* type);
    };
	    
	class Window
	{
	public:
		Window(int width, int height, const string& title, Window* sharingWindow = nullptr);
		~Window();

		bool ShouldClose();
		void MakeCurrent();
		void ProcessEvents();
		void Update(float timeDelta);
		void SwapBuffers();

		inline GLFWwindow* GetGLFWWindow() { return glfwWindow; }
		inline void UseVSync(bool use) { vSync = use; glfwSwapInterval(use ? 1 : 0); }
		inline bool IsUsingVSync() const { return vSync; }

		inline ImGuiIO* GetIO() { return io; }
	private:
		GLFWwindow* glfwWindow = nullptr;
		bool vSync = true;

		ImGuiIO* io = nullptr;
	};

	class Application
	{
	public:
		Application(int width = 1024, int height = 768, const string& windowTitle = "BlockA");
		~Application();

		void OnInitialize(function<void()> onInitialize);
		void OnUpdate(function<void(float)> onUpdate);
		void OnTerminate(function<void()> onTerminate);

		void Run();

		inline const string& GetResourceRoot() const { return resourceRoot; }
		inline void SetResourceRoot(const string& root) { resourceRoot = root; }

	protected:

	private:
		function<void()> onInitializeFunction;
		function<void(float)> onUpdateFunction;
		function<void()> onTerminateFunction;

		Window* window = nullptr;

		string resourceRoot;
	};
}