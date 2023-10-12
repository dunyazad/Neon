#include <Neon/System/NeonEventSystem.h>

#include <Neon/NeonEntity.h>
#include <Neon/NeonScene.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/NeonWindow.h>

#include <Neon/Component/NeonCamera.h>
#include <Neon/Component/NeonLight.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonShader.h>
#include <Neon/Component/NeonTexture.h>
#include <Neon/Component/NeonTransform.h>

namespace Neon
{
	set<EventSystem*> EventSystem::instances;

	EventSystem::EventSystem(Scene* scene)
		: SystemBase(scene)
	{
		if (0 == instances.size())
		{
			//struct {
			//	GLFWwindowposfun          pos;
			//	GLFWwindowsizefun         size;
			//	GLFWwindowclosefun        close;
			//	GLFWwindowrefreshfun      refresh;
			//	GLFWwindowfocusfun        focus;
			//	GLFWwindowiconifyfun      iconify;
			//	GLFWwindowmaximizefun     maximize;
			//	GLFWframebuffersizefun    fbsize;
			//	GLFWwindowcontentscalefun scale;
			//	GLFWmousebuttonfun        mouseButton;
			//	GLFWcursorposfun          cursorPos;
			//	GLFWcursorenterfun        cursorEnter;
			//	GLFWscrollfun             scroll;
			//	GLFWkeyfun                key;
			//	GLFWcharfun               character;
			//	GLFWcharmodsfun           charmods;
			//	GLFWdropfun               drop;
			//} callbacks;

			glfwSetKeyCallback(scene->GetWindow()->GetGLFWWindow(), EventSystem::KeyCallback);
			glfwSetMouseButtonCallback(scene->GetWindow()->GetGLFWWindow(), EventSystem::MouseButtonCallback);
			glfwSetCursorPosCallback(scene->GetWindow()->GetGLFWWindow(), EventSystem::CursorPosCallback);
			glfwSetScrollCallback(scene->GetWindow()->GetGLFWWindow(), EventSystem::ScrollCallback);
		}

		instances.insert(this);
	}

	EventSystem::~EventSystem()
	{
	}

	void EventSystem::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		for (auto& instance : instances)
		{
			instance->OnKeyEvent(window, key, scancode, action, mods);
		}
	}

	void EventSystem::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		for (auto& instance : instances)
		{
			instance->OnMouseButtonEvent(window, button, action, mods);
		}
	}

	void EventSystem::CursorPosCallback(GLFWwindow* window, double xpos, double ypos)
	{
		for (auto& instance : instances)
		{
			instance->OnCursorPosEvent(window, xpos, ypos);
		}
	}

	void EventSystem::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		for (auto& instance : instances)
		{
			instance->OnScrollEvent(window, xoffset, yoffset);
		}
	}

	void EventSystem::OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		for (auto& kvp : scene->GetEntities())
		{
			kvp.second->OnKeyEvent(window, key, scancode, action, mods);
		}
	}

	void EventSystem::OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods)
	{
		auto now = glfwGetTime();

		bool doubleClicked = false;
		if (action == GLFW_RELEASE)
		{
			if (button == GLFW_MOUSE_BUTTON_1)
			{
				auto delta = now - lastLButtonReleaseTime;
				if (delta < doubleClickInterval)
				{
					doubleClicked = true;
				}

				lastLButtonReleaseTime = now;
			}
			else if (button == GLFW_MOUSE_BUTTON_2)
			{
				auto delta = now - lastRButtonReleaseTime;
				if (delta < doubleClickInterval)
				{
					doubleClicked = true;
				}

				lastRButtonReleaseTime = now;
			}
			else if (button == GLFW_MOUSE_BUTTON_3)
			{
				auto delta = now - lastMButtonReleaseTime;
				if (delta < doubleClickInterval)
				{
					doubleClicked = true;
				}

				lastMButtonReleaseTime = now;
			}
		}

		for (auto& kvp : scene->GetEntities())
		{
			kvp.second->OnMouseButtonEvent(window, button, doubleClicked ? GLFW_DOUBLE_ACTION : action, mods);
		}
	}

	void EventSystem::OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos)
	{
		for (auto& kvp : scene->GetEntities())
		{
			kvp.second->OnCursorPosEvent(window, xpos, ypos);
		}
	}

	void EventSystem::OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset)
	{
		for (auto& kvp : scene->GetEntities())
		{
			kvp.second->OnScrollEvent(window, xoffset, yoffset);
		}
	}
}
