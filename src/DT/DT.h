#pragma once

#include <Neon/NeonCommon.h>

namespace DT
{
	template<class T>
	typename std::enable_if<std::is_same<T, float>::value, bool>::type
		almost_equal(T x, T y, int ulp = 2)
	{
		return fabsf(x - y) <= std::numeric_limits<float>::epsilon() * fabsf(x + y) * static_cast<float>(ulp)
			|| fabsf(x - y) < std::numeric_limits<float>::min();
	}

	template<class T>
	typename std::enable_if<std::is_same<T, double>::value, bool>::type
		almost_equal(T x, T y, int ulp = 2)
	{
		return fabs(x - y) <= std::numeric_limits<double>::epsilon() * fabs(x + y) * static_cast<double>(ulp)
			|| fabs(x - y) < std::numeric_limits<double>::min();
	}

	template<typename T>
	struct Vector2
	{
		using Type = T;
		Vector2() : x(0.0), y(0.0) {}
		Vector2(const Vector2<T>& v) : x(v.x), y(v.y) {}
		Vector2(Vector2<T>&& o) : x(o.x), y(o.y) {}
		Vector2(const T vx, const T vy) : x(vx), y(vy) {}

		T dist2(const Vector2<T>& v) const
		{
			const T dx = x - v.x;
			const T dy = y - v.y;
			return dx * dx + dy * dy;
		}

		T dist(const Vector2<T>& v) const
		{
			return hypotf(x - v.x, y - v.y);
		}

		//template<>
		//double
		//	Vector2<double>::dist(const Vector2<double>& v) const
		//{
		//	return hypot(x - v.x, y - v.y);
		//}

		T norm2() const
		{
			return x * x + y * y;
		}

		Vector2& operator=(const Vector2<T>&) = default;
		Vector2& operator=(Vector2&&) = default;
		bool operator ==(const Vector2<T>& v) const
		{
			return (this->x == v.x) && (this->y == v.y);
		}

		template<typename U>
		friend std::ostream& operator <<(std::ostream& str, const Vector2<U>& v)
		{
			return str << "Point x: " << v.x << " y: " << v.y;
		}

		T x;
		T y;

		static_assert(std::is_floating_point<Vector2<T>::Type>::value,
			"Type must be floating-point");
	};

	template<typename T>
	bool almost_equal(const Vector2<T>& v1, const Vector2<T>& v2)
	{
		return almost_equal(v1.x, v2.x) && almost_equal(v1.y, v2.y);
	}

	template<typename T>
	struct Edge
	{
		using Type = T;
		using VertexType = Vector2<Type>;

		Edge() = default;
		Edge(const Edge&) = default;
		Edge(Edge&&) = default;
		Edge(const VertexType& v1, const VertexType& v2) : v(&v1), w(&v2) {}

		Edge& operator=(const Edge&) = default;
		Edge& operator=(Edge&&) = default;
		bool operator ==(const Edge& e) const
		{
			return (*(this->v) == *e.v && *(this->w) == *e.w) ||
				(*(this->v) == *e.w && *(this->w) == *e.v);
		}

		template<typename U>
		friend std::ostream& operator <<(std::ostream& str, const Edge<U>& e)
		{
			return str << "Edge " << *e.v << ", " << *e.w;
		}

		const VertexType* v;
		const VertexType* w;
		bool isBad = false;

		static_assert(std::is_floating_point<Edge<T>::Type>::value,
			"Type must be floating-point");
	};

	template<typename T>
	bool
		almost_equal(const Edge<T>& e1, const Edge<T>& e2)
	{
		return	(almost_equal(*e1.v, *e2.v) && almost_equal(*e1.w, *e2.w)) ||
			(almost_equal(*e1.v, *e2.w) && almost_equal(*e1.w, *e2.v));
	}

	template<typename T>
	struct Triangle
	{
		using Type = T;
		using VertexType = Vector2<Type>;
		using EdgeType = Edge<Type>;

		Triangle() = default;
		Triangle(const Triangle&) = default;
		Triangle(Triangle&&) = default;
		Triangle(const VertexType& v1, const VertexType& v2, const VertexType& v3) : a(&v1), b(&v2), c(&v3) {}

		bool containsVertex(const VertexType& v) const
		{
			// return p1 == v || p2 == v || p3 == v;
			return almost_equal(*a, v) || almost_equal(*b, v) || almost_equal(*c, v);
		}

		bool circumCircleContains(const VertexType& v) const
		{
			const T ab = a->norm2();
			const T cd = b->norm2();
			const T ef = c->norm2();

			const T ax = a->x;
			const T ay = a->y;
			const T bx = b->x;
			const T by = b->y;
			const T cx = c->x;
			const T cy = c->y;

			const T circum_x = (ab * (cy - by) + cd * (ay - cy) + ef * (by - ay)) / (ax * (cy - by) + bx * (ay - cy) + cx * (by - ay));
			const T circum_y = (ab * (cx - bx) + cd * (ax - cx) + ef * (bx - ax)) / (ay * (cx - bx) + by * (ax - cx) + cy * (bx - ax));

			const VertexType circum(circum_x / 2, circum_y / 2);
			const T circum_radius = a->dist2(circum);
			const T dist = v.dist2(circum);
			return dist <= circum_radius;
		}

		Triangle& operator=(const Triangle&) = default;
		Triangle& operator=(Triangle&&) = default;
		bool operator ==(const Triangle& t) const
		{
			return	(*this->a == *t.a || *this->a == *t.b || *this->a == *t.c) &&
				(*this->b == *t.a || *this->b == *t.b || *this->b == *t.c) &&
				(*this->c == *t.a || *this->c == *t.b || *this->c == *t.c);
		}

		template<typename U>
		friend std::ostream& operator <<(std::ostream& str, const Triangle<U>& t)
		{
			return str << "Triangle:" << "\n\t" <<
				*t.a << "\n\t" <<
				*t.b << "\n\t" <<
				*t.c << '\n';
		}

		const VertexType* a;
		const VertexType* b;
		const VertexType* c;
		bool isBad = false;

		static_assert(std::is_floating_point<Triangle<T>::Type>::value,
			"Type must be floating-point");
	};

	template<typename T>
	bool almost_equal(const Triangle<T>& t1, const Triangle<T>& t2)
	{
		return	(almost_equal(*t1.a, *t2.a) || almost_equal(*t1.a, *t2.b) || almost_equal(*t1.a, *t2.c)) &&
			(almost_equal(*t1.b, *t2.a) || almost_equal(*t1.b, *t2.b) || almost_equal(*t1.b, *t2.c)) &&
			(almost_equal(*t1.c, *t2.a) || almost_equal(*t1.c, *t2.b) || almost_equal(*t1.c, *t2.c));
	}

	template<typename T>
	class Delaunay
	{
		using Type = T;
		using VertexType = Vector2<Type>;
		using EdgeType = Edge<Type>;
		using TriangleType = Triangle<Type>;

		static_assert(std::is_floating_point<Delaunay<T>::Type>::value, "Type must be floating-point");

		std::vector<TriangleType> _triangles;
		std::vector<EdgeType> _edges;
		std::vector<VertexType> _vertices;

	public:

		Delaunay() = default;
		Delaunay(const Delaunay&) = delete;
		Delaunay(Delaunay&&) = delete;

		const std::vector<TriangleType>& triangulate(std::vector<VertexType>& vertices)
		{
			// Store the vertices locally
			_vertices = vertices;

			// Determinate the super triangle
			T minX = vertices[0].x;
			T minY = vertices[0].y;
			T maxX = minX;
			T maxY = minY;

			for (std::size_t i = 0; i < vertices.size(); ++i)
			{
				if (vertices[i].x < minX) minX = vertices[i].x;
				if (vertices[i].y < minY) minY = vertices[i].y;
				if (vertices[i].x > maxX) maxX = vertices[i].x;
				if (vertices[i].y > maxY) maxY = vertices[i].y;
			}

			const T dx = maxX - minX;
			const T dy = maxY - minY;
			const T deltaMax = std::max(dx, dy);
			const T midx = (minX + maxX) / 2;
			const T midy = (minY + maxY) / 2;

			const VertexType p1(midx - 20 * deltaMax, midy - deltaMax);
			const VertexType p2(midx, midy + 20 * deltaMax);
			const VertexType p3(midx + 20 * deltaMax, midy - deltaMax);

			// Create a list of triangles, and add the supertriangle in it
			_triangles.push_back(TriangleType(p1, p2, p3));

			for (auto p = begin(vertices); p != end(vertices); p++)
			{
				std::vector<EdgeType> polygon;

				for (auto& t : _triangles)
				{
					if (t.circumCircleContains(*p))
					{
						t.isBad = true;
						polygon.push_back(Edge<T>{*t.a, * t.b});
						polygon.push_back(Edge<T>{*t.b, * t.c});
						polygon.push_back(Edge<T>{*t.c, * t.a});
					}
				}

				_triangles.erase(std::remove_if(begin(_triangles), end(_triangles), [](TriangleType& t) {
					return t.isBad;
					}), end(_triangles));

				for (auto e1 = begin(polygon); e1 != end(polygon); ++e1)
				{
					for (auto e2 = e1 + 1; e2 != end(polygon); ++e2)
					{
						if (almost_equal(*e1, *e2))
						{
							e1->isBad = true;
							e2->isBad = true;
						}
					}
				}

				polygon.erase(std::remove_if(begin(polygon), end(polygon), [](EdgeType& e) {
					return e.isBad;
					}), end(polygon));

				for (const auto e : polygon)
					_triangles.push_back(TriangleType(*e.v, *e.w, *p));

			}

			_triangles.erase(std::remove_if(begin(_triangles), end(_triangles), [p1, p2, p3](TriangleType& t) {
				return t.containsVertex(p1) || t.containsVertex(p2) || t.containsVertex(p3);
				}), end(_triangles));

			for (const auto t : _triangles)
			{
				_edges.push_back(Edge<T>{*t.a, * t.b});
				_edges.push_back(Edge<T>{*t.b, * t.c});
				_edges.push_back(Edge<T>{*t.c, * t.a});
			}

			return _triangles;
		}

		const std::vector<TriangleType>& getTriangles() const
		{
			return _triangles;
		}

		const std::vector<EdgeType>& getEdges() const
		{
			return _edges;
		}

		const std::vector<VertexType>& getVertices() const
		{
			return _vertices;
		}

		Delaunay& operator=(const Delaunay&) = delete;
		Delaunay& operator=(Delaunay&&) = delete;
	};
}