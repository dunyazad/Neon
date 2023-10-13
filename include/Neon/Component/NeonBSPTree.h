#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Mesh;

	class BSPTree : public ComponentBase
	{
	public:
		BSPTree(const string& name, Mesh* mesh);
		virtual ~BSPTree();
	};
}
