#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonWindow.h>
#include <Neon/NeonVertexArrayObject.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/NeonImage.h>
#include <Neon/NeonFrameBufferObject.h>

#include <Neon/NeonEntity.h>

#include <Neon/Component/NeonCamera.h>
#include <Neon/Component/NeonComponent.h>
#include <Neon/Component/NeonLight.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonShader.h>
#include <Neon/Component/NeonTexture.h>
#include <Neon/Component/NeonTransform.h>

#include <Neon/NeonScene.h>
#include <Neon/NeonSystem.h>

namespace Neon
{
	class Scene;

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

		Scene* CreateScene(const string& name);
		Scene* GetScene(const string& name);

	private:
		function<void()> onInitializeFunction;
		function<void(float, float)> onUpdateFunction;
		function<void()> onTerminateFunction;

		Window* window = nullptr;

		string resourceRoot;

		map<string, Scene*> scenes;
	};
}