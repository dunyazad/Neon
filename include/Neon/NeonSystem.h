#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class Scene;

	class SystemBase
	{
	public:
		SystemBase(Scene* scene);
		~SystemBase();

	protected:
		Scene* scene;
	};

	class TransformUpdateSystem : public SystemBase
	{
	public:
		TransformUpdateSystem(Scene* scene);
		~TransformUpdateSystem();

		void Frame(float now, float timeDelta);

	private:

	};

	class RenderSystem : public SystemBase
	{
	public:
		RenderSystem(Scene* scene);
		~RenderSystem();

		void Frame(float now, float timeDelta);

	private:

	};

	class EventSystem : public SystemBase
	{
	public:
		EventSystem(Scene* scene);
		~EventSystem();

		void Event();

	};
}
