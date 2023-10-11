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
		absoluteTransform = glm::mat4_cast(rotation);
		absoluteTransform = glm::translate(absoluteTransform, position);

		ComponentBase::OnUpdate(now, timeDelta);
	}
}
