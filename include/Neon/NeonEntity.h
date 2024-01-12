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

		template<typename T>
		void AddComponent(T* component)
		{
			if (nullptr != component)
			{
				components[&typeid(T)].push_back(component);
			}
		}

		template<typename T>
		vector<ComponentBase*> GetComponents()
		{
			return components[&typeid(T)];
		}

		template<typename T>
		T* GetComponent()
		{
			auto cv = components[&typeid(T)];
			if (0 == cv.size()) return nullptr;
			else return (T*)cv[0];
		}

		template<typename T>
		T* GetComponent(int index)
		{
			auto cv = components[&typeid(T)];
			if (0 == cv.size()) return nullptr;
			else if (index > cv.size() - 1) return nullptr;
			else return (T*)cv[index];
		}

		inline const map<const type_info*, vector<ComponentBase*>>& GetAllComponents() const { return components; }

		inline Scene* GetScene() const { return scene; }

		bool IsActive() const { return active; }
		void SetActive(bool isActive) { active = isActive; }
		void ToggleActive() { active = !active; }
		bool IsVisible() const { return visible; }
		void SetVisible(bool isVisible) { visible = isVisible; }
		void ToggleVisible() { visible = !visible; }

		virtual void OnKeyEvent(const KeyEvent& event);
		virtual void OnMouseButtonEvent(const MouseButtonEvent& event);
		virtual void OnCursorPosEvent(const CursorPosEvent& event);
		virtual void OnScrollEvent(const ScrollEvent& event);

	protected:
		Scene* scene;
		map<const type_info*, vector<ComponentBase*>> components;

		bool active = true;
		bool visible = true;
	};
}
