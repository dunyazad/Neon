#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/System/NeonSystem.h>

namespace Neon
{
	class Scene;
	
	class EventSystem : public SystemBase
	{
	public:
		EventSystem(Scene* scene);
		~EventSystem();
		
		static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
		static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
		static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

		void OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods);
		void OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods);
		void OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos);
		void OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset);

	protected:
		double doubleClickInterval = 0.5;
		double lastLButtonReleaseTime = 0.0;
		double lastRButtonReleaseTime = 0.0;
		double lastMButtonReleaseTime = 0.0;

		static set<EventSystem*> instances;
	};
}
