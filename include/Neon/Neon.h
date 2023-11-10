#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonImage.h>
#include <Neon/NeonTriangulator.h>
#include <Neon/NeonURL.h>
#include <Neon/NeonVertexArrayObject.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/NeonFrameBufferObject.h>
#include <Neon/NeonWindow.h>

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
#include <Neon/Component/NeonVETM.h>

#include <Neon/CUDA/CUDACommon.h>
#include <Neon/CUDA/CUDAMemoryPool.h>
#include <Neon/CUDA/CUDAMesh.h>
#include <Neon/CUDA/CUDAList.h>

#include <Neon/Component/SpatialPartitioning/NeonBSPTree.hpp>
#include <Neon/Component/SpatialPartitioning/NeonRegularGrid.h>
#include <Neon/Component/SpatialPartitioning/NeonSpatialHashing.h>

#include <Neon/System/NeonEventSystem.h>
#include <Neon/System/NeonRenderSystem.h>
#include <Neon/System/NeonSystem.h>

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

		inline Window* GetWindow() { return window; }

	private:
		function<void()> onInitializeFunction;
		function<void(double, double)> onUpdateFunction;
		function<void()> onTerminateFunction;

		Window* window = nullptr;

		string resourceRoot;

		map<string, Scene*> scenes;
	};
}
