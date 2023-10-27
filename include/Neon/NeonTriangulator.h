#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class Triangulator
	{
	public:
		Triangulator(vector<glm::vec3> points);
		~Triangulator();

		void Triangulate();

	private:
		vector<glm::vec3> points;
	};
}