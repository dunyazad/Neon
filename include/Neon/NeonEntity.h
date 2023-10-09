#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class ComponentBase;

	class Entity
	{
	public:
		Entity();
		~Entity();

	private:
		map<type_info, ComponentBase*> components;
	};
}
