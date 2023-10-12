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
}
