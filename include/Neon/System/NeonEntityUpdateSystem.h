#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/System/NeonSystem.h>

namespace Neon
{
	class Scene;

	class EntityUpdateSystem : public SystemBase
	{
	public:
		EntityUpdateSystem(Scene* scene);
		virtual ~EntityUpdateSystem();

		void Frame(double now, double timeDelta);
	};
}
