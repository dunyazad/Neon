#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class Triangulator
	{
	public:
		Triangulator();
		~Triangulator();

		vector<size_t> Triangulate(const vector<vector<glm::vec2>>& pointsList);
	};
}