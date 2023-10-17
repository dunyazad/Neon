#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Transform;

	class Light : public ComponentBase
	{
	public:
		Light(const string& name);
		virtual ~Light();

		virtual void OnUpdate(double now, double timeDelta);

		glm::vec3 position = glm::vec3(0.0f, 10.0f, 0.0f);
		glm::vec3 direction = glm::vec3(0.0f, -1.0f, 0.0f);
		glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);
	};
}