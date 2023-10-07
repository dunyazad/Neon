#pragma once

#include "NeonCommon.h"

namespace Neon
{
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
}