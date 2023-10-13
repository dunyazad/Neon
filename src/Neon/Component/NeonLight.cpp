#include <Neon/Component/NeonLight.h>
#include <Neon/Component/NeonTransform.h>

namespace Neon
{
	Light::Light(const string& name)
		: ComponentBase(name)
	{
	}

	Light::~Light()
	{
	}

	void Light::OnUpdate(float now, float timeDelta)
	{
		ComponentBase::OnUpdate(now, timeDelta);
	}
}
