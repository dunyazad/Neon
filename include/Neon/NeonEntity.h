#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class ComponentBase;
	class Scene;

	class Entity : public NeonObject
	{
	public:
		Entity(const string& name, Scene* scene);
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

		inline Scene* GetScene() const { return scene; }

		virtual void OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods);
		virtual void OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods);
		virtual void OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos);
		virtual void OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset);

	protected:
		Scene* scene;
		map<const type_info*, vector<ComponentBase*>> components;
	};
}
