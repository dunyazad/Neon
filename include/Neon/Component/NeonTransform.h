#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Transform : public ComponentBase
	{
	public:
		Transform(const string& name);
		virtual ~Transform();

		virtual void OnUpdate(float now, float timeDelta);

		glm::vec3 position = glm::zero<glm::vec3>();
		glm::quat rotation = glm::identity<glm::quat>();
		glm::mat4 absoluteTransform = glm::identity<glm::mat4>();

		Transform* parent = nullptr;
	};
}
