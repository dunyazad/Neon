#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonWindow.h>
#include <Neon/NeonVertexArrayObject.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/NeonImage.h>
#include <Neon/NeonFrameBufferObject.h>
#include <Neon/NeonURL.h>

#include <Neon/NeonDebugEntity.h>
#include <Neon/NeonEntity.h>
#include <Neon/NeonScene.h>

#include <Neon/Component/NeonCamera.h>
#include <Neon/Component/NeonCameraManipulator.h>
#include <Neon/Component/NeonComponent.h>
#include <Neon/Component/NeonLight.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonShader.h>
#include <Neon/Component/NeonTexture.h>
#include <Neon/Component/NeonTransform.h>

#include <Neon/Component/SpatialPartitioning/NeonBSPTree.hpp>
#include <Neon/Component/SpatialPartitioning/NeonRegularGrid.h>
#include <Neon/Component/SpatialPartitioning/NeonSpatialHashing.h>

#include <Neon/System/NeonEventSystem.h>
#include <Neon/System/NeonRenderSystem.h>
#include <Neon/System/NeonSystem.h>
#include <Neon/System/NeonTransformUpdateSystem.h>

namespace Neon
{
	class Scene;

	class Application
	{
	public:
		Application(int width = 1024, int height = 768, const string& windowTitle = "Neon");
		~Application();

		void OnInitialize(function<void()> onInitialize);
		void OnUpdate(function<void(double, double)> onUpdate);
		void OnTerminate(function<void()> onTerminate);

		void Run();

		Scene* CreateScene(const string& name);
		Scene* GetScene(const string& name);

	private:
		function<void()> onInitializeFunction;
		function<void(double, double)> onUpdateFunction;
		function<void()> onTerminateFunction;

		Window* window = nullptr;

		string resourceRoot;

		map<string, Scene*> scenes;
	};
}
