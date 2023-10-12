#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class ComponentBase : public NeonObject
	{
	public:
		ComponentBase(const string& name);
		virtual ~ComponentBase();
	};
}
