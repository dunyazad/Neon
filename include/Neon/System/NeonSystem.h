#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class Scene;

	class SystemBase
	{
	public:
		SystemBase(Scene* scene);
		virtual ~SystemBase();

	protected:
		Scene* scene;
	};
}
