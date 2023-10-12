#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/System/NeonSystem.h>

namespace Neon
{
	class Scene;

	class TransformUpdateSystem : public SystemBase
	{
	public:
		TransformUpdateSystem(Scene* scene);
		~TransformUpdateSystem();

		void Frame(float now, float timeDelta);
	};
}
