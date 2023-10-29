#include <Neon/NeonTriangulator.h>

namespace Neon
{
	Triangulator::Triangulator()
	{
	}

	Triangulator::~Triangulator()
	{
	}

	vector<GLuint> Triangulator::Triangulate(const vector<glm::vec2>& points)
	{
		vector<GLuint> result;

		auto rightmost = glm::zero<glm::vec2>();
		auto rightmostIndex = 0;
		for (size_t i = 0; i < points.size(); i++)
		{
			if (points[i].x > rightmost.x)
			{
				rightmostIndex = i;
			}
		}

		bool isClockWise = false;
		{
			auto pp = points[(rightmostIndex - 1) % points.size()];
			auto cp = points[rightmostIndex];
			auto np = points[(rightmostIndex + 1) % points.size()];
			isClockWise = glm::isClockwise(pp, cp, np);
		}

		
		set<pair<float, int>> angleInfos;
		for (size_t i = 0; i < points.size(); i++)
		{
			auto pp = points[(i - 1) % points.size()];
			auto cp = points[i];
			auto np = points[(i + 1) % points.size()];
			auto dp = glm::normalize(pp - cp);
			auto dn = glm::normalize(np - cp);
			//auto n = glm::normalize(glm::cross(glm::normalize(np - cp), glm::normalize(pp - cp)));
			//cout << n << endl;
			auto iscw = glm::isClockwise(pp, cp, np);

			auto angle = glm::angle(dp, dn);
			if (iscw != isClockWise)
			{
				angle = glm::pi<float>() * 2.0f - angle;
			}

			if (glm::degrees(angle) > 180)
			{
				cout << glm::degrees(angle) << endl;
			}

			angleInfos.insert(make_pair(angle, i));
		}

		for (auto& kvp : angleInfos)
		{
			cout << "[" << kvp.first << "] " << kvp.second << endl;
		}

		return result;
	}
}
