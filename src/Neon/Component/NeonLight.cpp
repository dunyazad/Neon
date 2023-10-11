#include <Neon/Component/NeonLight.h>

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
