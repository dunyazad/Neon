#include <Neon/Neon.h>
#include <Neon/NeonScene.h>

#include <Neon/System/NeonEventSystem.h>
#include <Neon/System/NeonRenderSystem.h>
#include <Neon/System/NeonSystem.h>
#include <Neon/System/NeonTransformUpdateSystem.h>

namespace Neon
{
	Application::Application(int width, int height, const string& windowTitle)
	{
		resourceRoot = std::filesystem::current_path().string();

		window = new Window(width, height, windowTitle);
	}

	Application::~Application()
	{
	}

	void Application::OnInitialize(function<void()> onInitialize)
	{
		onInitializeFunction = onInitialize;
	}

	void Application::OnUpdate(function<void(float, float)> onUpdate)
	{
		onUpdateFunction = onUpdate;
	}

	void Application::OnTerminate(function<void()> onTerminate)
	{
		onTerminateFunction = onTerminate;
	}

	Scene* Application::CreateScene(const string& name)
	{
		if (0 == scenes.count(name))
			scenes[name] = new Scene(name, window);
		return scenes[name];
	}

	Scene* Application::GetScene(const string& name)
	{
		if (0 == scenes.count(name))
			return nullptr;
		else return scenes[name];
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

		auto lastTime = glfwGetTime() * 1000.0;

		while (appFinisihed == false)
		{
			auto now = glfwGetTime() * 1000.0;
			auto timeDelta = now - lastTime;

			appFinisihed = true;
			if (window->ShouldClose() == false)
			{
				appFinisihed = false;

				//window->MakeCurrent();
				window->ProcessEvents();

				if (onUpdateFunction != nullptr)
				{
					onUpdateFunction((float)now, (float)timeDelta);
				}

				for (auto& kvp : scenes)
				{
					kvp.second->Frame((float)now, (float)timeDelta);
				}

#pragma region imgui
				// Start the Dear ImGui frame
				ImGui_ImplOpenGL3_NewFrame();
				ImGui_ImplGlfw_NewFrame();
				ImGui::NewFrame();

				// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
				//if (show_demo_window)
				//    ImGui::ShowDemoWindow(&show_demo_window);

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
#pragma endregion

			}

			window->SwapBuffers();

			lastTime = now;
		}

		if (onTerminateFunction != nullptr)
		{
			onTerminateFunction();
		}

		for (auto& kvp : scenes)
		{
			SAFE_DELETE(kvp.second);
		}

		glfwTerminate();
	}
}
