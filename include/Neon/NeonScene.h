#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonSystem.h>

namespace Neon
{
	class Entity;
	class ComponentBase;
	class Camera;

	class Scene
	{
	public:
		Scene(const string& name);
		~Scene();

		Entity* GetEntity(const string& name);
		Entity* CreateEntity(const string& name);

		template <class T, typename... Args>
		T* CreateComponent(const std::string& name, Args... args)
		{
			if (0 == componentNameMapping.count(name))
			{
				auto key = &typeid(T);
				auto component = new T(name, args...);
				components[key].push_back(component);
				componentNameMapping[name] = component;
				return component;
			}
			else
			{
				return (T*)componentNameMapping[name];
			}
		}

		template<class T> T* CreateComponent(const string& name)
		{
			if (0 == componentNameMapping.count(name))
			{
				auto key = &typeid(T);
				auto component = new T(name);
				components[key].push_back(component);
				componentNameMapping[name] = component;
				return component;
			}
			else
			{
				return (T*)componentNameMapping[name];
			}
		}

		template <class T, typename FirstArg, typename... RestArgs>
		T* CreateComponent(const std::string& name, FirstArg firstArg, RestArgs... restArgs) {
			if (0 == componentNameMapping.count(name))
			{
				auto key = &typeid(T);
				auto component = new T(name, firstArg, restArgs...);
				components[key].push_back(component);
				componentNameMapping[name] = component;
				return component;
			}
			else
			{
				return (T*)componentNameMapping[name];
			}
		}

		template<class T>
		T* GetComponent(const string& name)
		{
			if (0 != componentNameMapping.count(name))
			{
				return (T*)componentNameMapping[name];
			}
			else
			{
				return nullptr;
			}
		}

		inline const map<string, Entity*>& GetEntities() const { return entities; }
		template<class T> vector<ComponentBase*>& GetComponents() { return components[&typeid(T)]; }

		void Frame(float now, float timeDelta);

		inline Camera* GetMainCamera() { return mainCamera; }
		inline void SetMainCamera(Camera* camera) { mainCamera = camera; }

	private:
		string name;
		map<string, Entity*> entities;
		map<const type_info*, vector<ComponentBase*>> components;
		map<string, ComponentBase*> componentNameMapping;

		Camera* mainCamera = nullptr;

		TransformUpdateSystem transformUpdateSystem;
		RenderSystem renderSystem;
	};
}
