#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class ComponentBase
	{
	public:
		ComponentBase(const string& name);
		~ComponentBase();

	private:
		string name;
	};
}
