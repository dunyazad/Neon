#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonWindow.h>
#include <Neon/NeonVertexArrayObject.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/NeonImage.h>
#include <Neon/NeonFrameBufferObject.h>

#include <Neon/NeonEntity.h>

#include <Neon/Component/NeonComponent.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonShader.h>
#include <Neon/Component/NeonTexture.h>
#include <Neon/Component/NeonTransform.h>

#include <Neon/NeonSystem.h>

namespace Neon
{
	//bool operator<(const type_info& a, const type_info& b)
	//{
	//	return int(&a) < int(&b);
	//}

	//bool operator<(const type_info* a, const type_info* b)
	//{
	//	return int(a) < int(b);
	//}

	class Application
	{
	public:
		Application(int width = 1024, int height = 768, const string& windowTitle = "Neon");
		~Application();

		void OnInitialize(function<void()> onInitialize);
		void OnUpdate(function<void(float, float)> onUpdate);
		void OnTerminate(function<void()> onTerminate);

		void Run();

		inline const string& GetResourceRoot() const { return resourceRoot; }
		inline void SetResourceRoot(const string& root) { resourceRoot = root; }

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


		inline ComponentBase* GetComponent(const string& name)
		{
			if (0 != componentNameMapping.count(name))
			{
				return componentNameMapping[name];
			}
			else
			{
				return nullptr;
			}
		}

		inline const map<string, Entity*>& GetEntities() const { return entities; }
		template<class T> vector<ComponentBase*>& GetComponents() { return components[&typeid(T)]; }

	private:
		function<void()> onInitializeFunction;
		function<void(float, float)> onUpdateFunction;
		function<void()> onTerminateFunction;

		Window* window = nullptr;

		string resourceRoot;

		map<string, Entity*> entities;
		map<const type_info*, vector<ComponentBase*>> components;
		map<string, ComponentBase*> componentNameMapping;

		TransformUpdateSystem transformUpdateSystem;
		RenderSystem renderSystem;
	};
}
