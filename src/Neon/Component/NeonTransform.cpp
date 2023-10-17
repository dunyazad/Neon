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

	void Transform::OnUpdate(double now, double timeDelta)
	{
		absoluteTransform = glm::mat4_cast(rotation);
		absoluteTransform = glm::translate(absoluteTransform, position);

		if (nullptr != parent)
		{
			absoluteTransform = parent->absoluteTransform * absoluteTransform;
		}

		ComponentBase::OnUpdate(now, timeDelta);
	}
}
