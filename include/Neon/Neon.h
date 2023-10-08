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

namespace Neon
{
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

	protected:

	private:
		function<void()> onInitializeFunction;
		function<void(float)> onUpdateFunction;
		function<void()> onTerminateFunction;

		Window* window = nullptr;

		string resourceRoot;
	};
}
