#include <Neon/NeonTransform.h>

namespace Neon
{
	Transform::Transform(const string& name)
		: ComponentBase(name)
	{
	}

	Transform::~Transform()
	{
	}

	void Transform::OnUpdate(float timeDelta)
	{
		for (auto& callback : updateCallbacks)
		{
			callback(timeDelta);
		}
	}
}
