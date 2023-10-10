#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonComponent.h>

namespace Neon
{
	class Transform : public ComponentBase
	{
	public:
		Transform(const string& name);
		~Transform();

		void OnUpdate(float timeDelta);

	private:
		set<function<void(float)>> updateCallbacks;

		glm::vec3 position = glm::zero<glm::vec3>();
		glm::quat rotation = glm::identity<glm::quat>();
	};
}
