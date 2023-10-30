#include <Neon/NeonTriangulator.h>

#include <mapbox/earcut.hpp>

namespace Neon
{
	Triangulator::Triangulator()
	{
	}

	Triangulator::~Triangulator()
	{
	}

	vector<size_t> Triangulator::Triangulate(const vector<glm::vec2>& points)
	{
		vector<size_t> result;

		using Point = std::array<float, 2>;
		vector<vector<Point>> polygon;

		vector<Point> contour;
		for (auto& p : points)
		{
			contour.push_back({ p.x, p.y });
		}
		polygon.push_back(contour);

		vector<size_t> indices = mapbox::earcut<size_t>(polygon);

		return indices;

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

		
		set<pair<float, size_t>> angleInfos;
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

			angleInfos.insert(make_pair(angle, i));
		}

		while (4 < angleInfos.size())
		{
			auto current = (*angleInfos.begin());
			auto ci = current.second;

			result.push_back((ci - 1) % points.size());
			result.push_back((ci) % points.size());
			result.push_back((ci + 1) % points.size());

			angleInfos.erase(angleInfos.begin());

			for (auto i = angleInfos.begin(); i != angleInfos.end();)
			{
				if ((*i).second == ci - 1)
				{
					i = angleInfos.erase(i);
				}
				else if ((*i).second == ci + 1)
				{
					i = angleInfos.erase(i);
				}
				else
				{
					++i;
				}
			}

			{
				auto pp = points[(ci - 2) % points.size()];
				auto cp = points[(ci - 1) % points.size()];
				auto np = points[(ci + 1) % points.size()];

				auto dp = glm::normalize(pp - cp);
				auto dn = glm::normalize(np - cp);

				auto iscw = glm::isClockwise(pp, cp, np);

				auto angle = glm::angle(dp, dn);
				if (iscw != isClockWise)
				{
					angle = glm::pi<float>() * 2.0f - angle;
				}

				angleInfos.insert(make_pair(angle, ci - 1));
			}

			{
				auto pp = points[(ci - 1) % points.size()];
				auto cp = points[(ci + 1) % points.size()];
				auto np = points[(ci + 2) % points.size()];

				auto dp = glm::normalize(pp - cp);
				auto dn = glm::normalize(np - cp);

				auto iscw = glm::isClockwise(pp, cp, np);

				auto angle = glm::angle(dp, dn);
				if (iscw != isClockWise)
				{
					angle = glm::pi<float>() * 2.0f - angle;
				}

				angleInfos.insert(make_pair(angle, ci + 1));
			}

			//auto pp = points[(ci - 1) % points.size()];
			//auto cp = points[ci];
			//auto np = points[(ci + 1) % points.size()];
		}


		int i = 0;
		for (auto& kvp : angleInfos)
		{
			cout << "[" << i++ << "] " << glm::degrees(kvp.first) << ", " << kvp.second << endl;
		}

		return result;
	}
}
