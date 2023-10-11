#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Entity;

	class EventSubscriber : public ComponentBase
	{
	public:
		EventSubscriber(const string& name, Entity* entity);
		virtual ~EventSubscriber();

		void OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods);
		void OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods);
		void OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos);
		void OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset);

		inline void SetKeyEventCallback(function<void(GLFWwindow*, int, int, int, int)> callback) { keyEventCallback = callback; }
		inline void SetMouseButtonEventCallback(function<void(GLFWwindow*, int, int, int)> callback) { mouseButtonEventCallback = callback; }
		inline void SetCursorPosEventCallback(function<void(GLFWwindow*, double, double)> callback) { cursorPosEventCallback = callback; }
		inline void SetScrollEventCallback(function<void(GLFWwindow*, double, double)> callback) { scrollEventCallback = callback;  }

	protected:
		Entity* entity;

		function<void(GLFWwindow*, int, int, int, int)> keyEventCallback;
		function<void(GLFWwindow*, int, int, int)> mouseButtonEventCallback;
		function<void(GLFWwindow*, double, double)> cursorPosEventCallback;
		function<void(GLFWwindow*, double, double)> scrollEventCallback;
	};
}
