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

	vector<size_t> Triangulator::Triangulate(const vector<vector<glm::vec2>>& pointsList)
	{
		vector<size_t> result;

		using Point = std::array<float, 2>;
		vector<vector<Point>> polygon;

		for (auto& points : pointsList)
		{
			vector<Point> contour;
			for (auto& p : points)
			{
				contour.push_back({ p.x, p.y });
			}
			polygon.push_back(contour);
		}

		return mapbox::earcut<size_t>(polygon);
	}
}
