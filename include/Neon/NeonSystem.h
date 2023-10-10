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

	class TransformUpdateSystem : public SystemBase
	{
	public:
		TransformUpdateSystem(Application* application);
		~TransformUpdateSystem();

		void Frame(float now, float timeDelta);

	private:

	};

	class RenderSystem : public SystemBase
	{
	public:
		RenderSystem(Application* application);
		~RenderSystem();

		void Frame(float now, float timeDelta);

	private:

	};
}
