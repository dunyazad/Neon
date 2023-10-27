#include <Neon/NeonTriangulator.h>

namespace Neon
{
	Triangulator::Triangulator()
	{
	}

	Triangulator::~Triangulator()
	{
	}

	vector<GLuint> Triangulator::Triangulate(const vector<glm::vec3>& points)
	{
		vector<GLuint> result;

		for (size_t i = 0; i < points.size(); i++)
		{
			auto pp = points[(i - 1) % points.size()];
			auto cp = points[i];
			auto np = points[(i + 1) % points.size()];
			auto dp = glm::normalize(pp - cp);
			auto dn = glm::normalize(np - cp);
			//auto n = glm::normalize(glm::cross(glm::normalize(np - cp), glm::normalize(pp - cp)));
			//cout << n << endl;
			auto angle = glm::angle(dp, dn);
			if (glm::degrees(angle) < 178)
			{
				cout << glm::degrees(angle) << endl;
			}
		}

		return result;
	}
}
