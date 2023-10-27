#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class Triangulator
	{
	public:
		Triangulator();
		~Triangulator();

		vector<GLuint> Triangulate(const vector<glm::vec3>& points);
	};
}