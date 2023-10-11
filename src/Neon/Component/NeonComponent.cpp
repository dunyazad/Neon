#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	ComponentBase::ComponentBase(const string& name)
		: name(name)
	{
	}

	ComponentBase::~ComponentBase()
	{
	}

	void ComponentBase::OnUpdate(float now, float timeDelta)
	{
		for (auto& callback : updateCallbacks)
		{
			callback(now, timeDelta);
		}
	}
}
