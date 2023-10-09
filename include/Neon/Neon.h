#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonWindow.h>
#include <Neon/NeonShader.h>
#include <Neon/NeonVertexArrayObject.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/NeonImage.h>
#include <Neon/NeonTexture.h>
#include <Neon/NeonFrameBufferObject.h>
#include <Neon/NeonRenderData.h>

#include <Neon/NeonEntity.h>
#include <Neon/NeonComponent.h>
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
		void OnUpdate(function<void(float)> onUpdate);
		void OnTerminate(function<void()> onTerminate);

		void Run();

		inline const string& GetResourceRoot() const { return resourceRoot; }
		inline void SetResourceRoot(const string& root) { resourceRoot = root; }

		Entity* GetEntity(const string& name);
		Entity* CreateEntity(const string& name);

		template<class T> T* CreateComponent()
		{
			auto key = &typeid(T);
			auto component = new T;
			components[key].push_back(component);
			return component;
		}

		inline const map<string, Entity*>& GetEntities() const { return entities; }
		template<class T> vector<ComponentBase*>& GetComponents() { return components[&typeid(T)]; }

	protected:

	private:
		function<void()> onInitializeFunction;
		function<void(float)> onUpdateFunction;
		function<void()> onTerminateFunction;

		Window* window = nullptr;

		string resourceRoot;

		map<string, Entity*> entities;
		map<const type_info*, vector<ComponentBase*>> components;

		RenderSystem renderSystem;
	};
}
