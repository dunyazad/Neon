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

	void Light::OnUpdate(double now, double timeDelta)
	{
		ComponentBase::OnUpdate(now, timeDelta);
	}
}
