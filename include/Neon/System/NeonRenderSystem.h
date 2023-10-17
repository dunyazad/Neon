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
		virtual ~RenderSystem();

		void Frame(double now, double timeDelta);

	private:

	};
}
