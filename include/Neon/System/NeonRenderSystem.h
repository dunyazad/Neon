#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/System/NeonSystem.h>

namespace Neon
{
	class Scene;
	
	class RenderSystem : public SystemBase
	{
	public:
		RenderSystem(Scene* scene);
		~RenderSystem();

		void Frame(float now, float timeDelta);

	private:

	};
}
