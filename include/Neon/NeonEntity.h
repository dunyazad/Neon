#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class ComponentBase;

	class Entity
	{
	public:
		Entity(const string& name);
		~Entity();

		template<class T>
		void AddComponent(T* component)
		{
			if (nullptr != component)
			{
				components[&typeid(T)].push_back(component);
			}
		}

		template<class T>
		vector<ComponentBase*> GetComponents()
		{
			return components[&typeid(T)];
		}

		template<class T>
		T* GetComponent(int index)
		{
			auto cv = components[&typeid(T)];
			if (0 == cv.size()) return nullptr;
			else if (index > cv.size() - 1) return nullptr;
			else return (T*)cv[index];
		}

		inline const string& GetName() const { return name; }

		void OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods);
		void OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods);
		void OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos);
		void OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset);

		inline void SetKeyEventCallback(function<void(GLFWwindow*, int, int, int, int)> callback) { keyEventCallback = callback; }
		inline void SetMouseButtonEventCallback(function<void(GLFWwindow*, int, int, int)> callback) { mouseButtonEventCallback = callback; }
		inline void SetCursorPosEventCallback(function<void(GLFWwindow*, double, double)> callback) { cursorPosEventCallback = callback; }
		inline void SetScrollEventCallback(function<void(GLFWwindow*, double, double)> callback) { scrollEventCallback = callback; }

	protected:
		string name;

		map<const type_info*, vector<ComponentBase*>> components;

		function<void(GLFWwindow*, int, int, int, int)> keyEventCallback;
		function<void(GLFWwindow*, int, int, int)> mouseButtonEventCallback;
		function<void(GLFWwindow*, double, double)> cursorPosEventCallback;
		function<void(GLFWwindow*, double, double)> scrollEventCallback;
	};
}
