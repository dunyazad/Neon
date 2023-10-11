#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class EventSubscriber : public ComponentBase
	{
	public:
		EventSubscriber(const string& name);
		virtual ~EventSubscriber();

	protected:
	};
}
