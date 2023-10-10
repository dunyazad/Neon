#include <Neon/Component/NeonTransform.h>

namespace Neon
{
	Transform::Transform(const string& name)
		: ComponentBase(name)
	{
	}

	Transform::~Transform()
	{
	}

	void Transform::OnUpdate(float now, float timeDelta)
	{
		for (auto& callback : updateCallbacks)
		{
			callback(now, timeDelta);
		}

		absoluteTransform = glm::mat4_cast(rotation);
		absoluteTransform = glm::translate(absoluteTransform, position);
	}
}
