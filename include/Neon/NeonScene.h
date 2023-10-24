#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/System/NeonSystem.h>
#include <Neon/System/NeonEventSystem.h>
#include <Neon/System/NeonRenderSystem.h>
#include <Neon/System/NeonTransformUpdateSystem.h>

namespace Neon
{
	class Window;
	class Entity;
	class DebugEntity;
	class ComponentBase;
	class Camera;
	class Light;

	class Scene
	{
	public:
		Scene(const string& name, Window* window);
		~Scene();

		Entity* GetEntity(const string& name);
		Entity* CreateEntity(const string& name);

		//DebugEntity* GetDebugEntity(const string& name);
		DebugEntity* CreateDebugEntity(const string& name);

		DebugEntity* Debug(const string& name);

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

		template<typename T> T* CreateComponent(const string& name)
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

		template<typename T>
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
		inline const vector<DebugEntity*>& GetDebugEntities() const { return debugEntities; }
		inline DebugEntity* GetDebugEntity(const string& name) { if (0 == debugEntityNameMap.count(name)) return nullptr; else return debugEntities[debugEntityNameMap[name]]; }
		template<typename T> vector<ComponentBase*>& GetComponents() { return components[&typeid(T)]; }

		void Frame(double now, double timeDelta);

		inline Window* GetWindow() { return window; }

		inline Camera* GetMainCamera() { return mainCamera; }
		inline void SetMainCamera(Camera* camera) { mainCamera = camera; }

		inline Light* GetMainLight() { return mainLight; }
		inline void SetMainLight(Light* light) { mainLight = light; }

	private:
		string name;
		bool active = true;
		Window* window;
		map<string, Entity*> entities;
		map<string, size_t> debugEntityNameMap;
		vector<DebugEntity*> debugEntities;
		map<const type_info*, vector<ComponentBase*>> components;
		map<string, ComponentBase*> componentNameMapping;

		Camera* mainCamera = nullptr;
		Light* mainLight = nullptr;

		TransformUpdateSystem transformUpdateSystem;
		RenderSystem renderSystem;
		EventSystem eventSystem;
	};
}
