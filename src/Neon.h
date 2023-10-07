#pragma once

#include "NeonCommon.h"
#include "NeonWindow.h"
#include "NeonShader.h"
#include "NeonVertexArrayObject.h"
#include "NeonVertexBufferObject.hpp"

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
