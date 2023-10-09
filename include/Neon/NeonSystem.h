#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class Application;

	class SystemBase
	{
	public:
		SystemBase(Application* application);
		~SystemBase();

	protected:
		Application* application;
	};

	class RenderSystem : public SystemBase
	{
	public:
		RenderSystem(Application* application);
		~RenderSystem();

		void Frame(float timeDelta);

	private:

	};
}
