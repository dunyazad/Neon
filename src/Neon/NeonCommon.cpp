#include <Neon/NeonCommon.h>

float Trimax(float a, float b, float c) {
	return std::max(std::max(a, b), c);
}

float Trimin(float a, float b, float c) {
	return std::min(std::min(a, b), c);
}

ostream& operator << (ostream& o, const glm::vec2& v)
{
	return o << v.x << " " << v.y;
}

ostream& operator << (ostream& o, const glm::vec3& v)
{
	return o << v.x << " " << v.y << " " << v.z;
}

ostream& operator << (ostream& o, const glm::vec4& v)
{
	return o << v.x << "\t" << v.y << "\t" << v.z << "\t" << v.w;
}

ostream& operator << (ostream& o, const glm::mat3& m)
{
	return o << m[0] << endl << m[1] << endl << m[2];
}

ostream& operator << (ostream& o, const glm::mat4& m)
{
	return o << m[0] << endl << m[1] << endl << m[2] << endl << m[3];
}

ostream& operator << (ostream& o, const glm::quat& q)
{
	return o << q.w << "\t" << q.x << "\t" << q.y << "\t" << q.z;
}

bool operator < (const glm::vec3& a, const glm::vec3& b)
{
	if (a.x < b.x) return true;
	else if (a.x > b.x) return false;
	else
	{
		if (a.y < b.y) return true;
		else if (a.y > b.y) return false;
		else
		{
			if (a.z < b.z) return true;
			else if (a.z > b.z) return false;
			else
			{
				return false;
			}
		}
	}
}

bool operator > (const glm::vec3& a, const glm::vec3& b)
{
	if (a.x > b.x) return true;
	else if (a.x < b.x) return false;
	else
	{
		if (a.y > b.y) return true;
		else if (a.y < b.y) return false;
		else
		{
			if (a.z > b.z) return true;
			else if (a.z < b.z) return false;
			else
			{
				return true;
			}
		}
	}
}

void _CheckGLError(const char* file, int line)
{
	GLenum err(glGetError());

	while (err != GL_NO_ERROR)
	{
		std::string error;
		switch (err)
		{
		case GL_INVALID_OPERATION:  error = "INVALID_OPERATION";      break;
		case GL_INVALID_ENUM:       error = "INVALID_ENUM";           break;
		case GL_INVALID_VALUE:      error = "INVALID_VALUE";          break;
		case GL_OUT_OF_MEMORY:      error = "OUT_OF_MEMORY";          break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:  error = "INVALID_FRAMEBUFFER_OPERATION";  break;
		}
		std::cout << "GL_" << error.c_str() << " - " << file << ":" << line << std::endl;
		err = glGetError();
	}

	return;
}

namespace Neon
{
	json Settings;

	AABB::AABB(const glm::vec3& minPoint, const glm::vec3& maxPoint)
		: xyz(minPoint), XYZ(maxPoint)
	{
		update();
	}

	void AABB::update()
	{
		xYz.x = xyz.x; xYz.y = XYZ.y; xYz.z = xyz.z;
		xyZ.x = xyz.x; xyZ.y = xyz.y; xyZ.z = XYZ.z;
		xYZ.x = xyz.x; xYZ.y = XYZ.y; xYZ.z = XYZ.z;
		Xyz.x = XYZ.x; Xyz.y = xyz.y; Xyz.z = xyz.z;
		XYz.x = XYZ.x; XYz.y = XYZ.y; XYz.z = xyz.z;
		XyZ.x = XYZ.x; XyZ.y = xyz.y; XyZ.z = XYZ.z;

		center.x = (xyz.x + XYZ.x) * float(0.5);
		center.y = (xyz.y + XYZ.y) * float(0.5);
		center.z = (xyz.z + XYZ.z) * float(0.5);
		extents.x = XYZ.x - center.x;
		extents.y = XYZ.y - center.y;
		extents.z = XYZ.z - center.z;
	}

	bool AABB::IntersectsRay(const Ray& ray, vector<glm::vec3>& intersections)
	{
		glm::vec3 t_min = {
			(xyz.x - ray.origin.x) / ray.direction.x,
			(xyz.y - ray.origin.y) / ray.direction.y,
			(xyz.z - ray.origin.z) / ray.direction.z
		};

		glm::vec3 t_max = {
			(XYZ.x - ray.origin.x) / ray.direction.x,
			(XYZ.y - ray.origin.y) / ray.direction.y,
			(XYZ.z - ray.origin.z) / ray.direction.z
		};

		glm::vec3 t_enter = {
			std::min(t_min.x, t_max.x),
			std::min(t_min.y, t_max.y),
			std::min(t_min.z, t_max.z)
		};

		glm::vec3 t_exit = {
			std::max(t_min.x, t_max.x),
			std::max(t_min.y, t_max.y),
			std::max(t_min.z, t_max.z)
		};

		auto max_t_enter = Trimax(t_enter.x, t_enter.y, t_enter.z);
		auto min_t_exit = Trimin(t_exit.x, t_exit.y, t_exit.z);

		if (max_t_enter <= min_t_exit) {
			if (max_t_enter <= min_t_exit) {
				intersections.push_back(ray.origin + ray.direction * max_t_enter);
			}
			if (min_t_exit >= 0) {
				intersections.push_back(ray.origin + ray.direction * min_t_exit);
			}
			return true;
		}

		return false;
	}

	bool AABB::IntersectsTriangle(const glm::vec3& tp0, const glm::vec3& tp1, const glm::vec3& tp2)
	{
		glm::vec3 v0 = { tp0.x - center.x, tp0.y - center.y, tp0.z - center.z };
		glm::vec3 v1 = { tp1.x - center.x, tp1.y - center.y, tp1.z - center.z };
		glm::vec3 v2 = { tp2.x - center.x, tp2.y - center.y, tp2.z - center.z };

		// Compute edge vectors for triangle
		glm::vec3 f0 = { tp1.x - tp0.x, tp1.y - tp0.y, tp1.z - tp0.z };
		glm::vec3 f1 = { tp2.x - tp1.x, tp2.y - tp1.y, tp2.z - tp1.z };
		glm::vec3 f2 = { tp0.x - tp2.x, tp0.y - tp2.y, tp0.z - tp2.z };

		//// region Test axes a00..a22 (category 3)

		// Test axis a00
		glm::vec3 a00 = { 0, -f0.z, f0.y };
		float p0 = v0.x * a00.x + v0.y * a00.y + v0.z * a00.z;
		float p1 = v1.x * a00.x + v1.y * a00.y + v1.z * a00.z;
		float p2 = v2.x * a00.x + v2.y * a00.y + v2.z * a00.z;
		float r = extents.y * std::fabs(f0.z) + extents.z * std::fabs(f0.y);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a01
		glm::vec3 a01 = { 0, -f1.z, f1.y };
		p0 = v0.x * a01.x + v0.y * a01.y + v0.z * a01.z;
		p1 = v1.x * a01.x + v1.y * a01.y + v1.z * a01.z;
		p2 = v2.x * a01.x + v2.y * a01.y + v2.z * a01.z;
		r = extents.y * std::fabs(f1.z) + extents.z * std::fabs(f1.y);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a02
		glm::vec3 a02 = { 0, -f2.z, f2.y };
		p0 = v0.x * a02.x + v0.y * a02.y + v0.z * a02.z;
		p1 = v1.x * a02.x + v1.y * a02.y + v1.z * a02.z;
		p2 = v2.x * a02.x + v2.y * a02.y + v2.z * a02.z;
		r = extents.y * std::fabs(f2.z) + extents.z * std::fabs(f2.y);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a10
		glm::vec3 a10 = { f0.z, 0, -f0.x };
		p0 = v0.x * a10.x + v0.y * a10.y + v0.z * a10.z;
		p1 = v1.x * a10.x + v1.y * a10.y + v1.z * a10.z;
		p2 = v2.x * a10.x + v2.y * a10.y + v2.z * a10.z;
		r = extents.x * std::fabs(f0.z) + extents.z * std::fabs(f0.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a11
		glm::vec3 a11 = { f1.z, 0, -f1.x };
		p0 = v0.x * a11.x + v0.y * a11.y + v0.z * a11.z;
		p1 = v1.x * a11.x + v1.y * a11.y + v1.z * a11.z;
		p2 = v2.x * a11.x + v2.y * a11.y + v2.z * a11.z;
		r = extents.x * std::fabs(f1.z) + extents.z * std::fabs(f1.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a12
		glm::vec3 a12 = { f2.z, 0, -f2.x };
		p0 = v0.x * a12.x + v0.y * a12.y + v0.z * a12.z;
		p1 = v1.x * a12.x + v1.y * a12.y + v1.z * a12.z;
		p2 = v2.x * a12.x + v2.y * a12.y + v2.z * a12.z;
		r = extents.x * std::fabs(f2.z) + extents.z * std::fabs(f2.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a20
		glm::vec3 a20 = { -f0.y, f0.x, 0 };
		p0 = v0.x * a20.x + v0.y * a20.y + v0.z * a20.z;
		p1 = v1.x * a20.x + v1.y * a20.y + v1.z * a20.z;
		p2 = v2.x * a20.x + v2.y * a20.y + v2.z * a20.z;
		r = extents.x * std::fabs(f0.y) + extents.y * std::fabs(f0.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a21
		glm::vec3 a21 = { -f1.y, f1.x, 0 };
		p0 = v0.x * a21.x + v0.y * a21.y + v0.z * a21.z;
		p1 = v1.x * a21.x + v1.y * a21.y + v1.z * a21.z;
		p2 = v2.x * a21.x + v2.y * a21.y + v2.z * a21.z;
		r = extents.x * std::fabs(f1.y) + extents.y * std::fabs(f1.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a22
		glm::vec3 a22 = { -f2.y, f2.x, 0 };
		p0 = v0.x * a22.x + v0.y * a22.y + v0.z * a22.z;
		p1 = v1.x * a22.x + v1.y * a22.y + v1.z * a22.z;
		p2 = v2.x * a22.x + v2.y * a22.y + v2.z * a22.z;
		r = extents.x * std::fabs(f2.y) + extents.y * std::fabs(f2.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		//// endregion

		//// region Test the three axes corresponding to the face normals of AABB b (category 1)

		// Exit if...
		// ... [-extents.X, extents.X] and [Min(v0.X,v1.X,v2.X), Max(v0.X,v1.X,v2.X)] do not overlap
		if (Trimax(v0.x, v1.x, v2.x) < -extents.x || Trimin(v0.x, v1.x, v2.x) > extents.x) {
			if (Trimax(v0.x, v1.x, v2.x) - (-extents.x) > FLT_EPSILON ||
				Trimin(v0.x, v1.x, v2.x) - (extents.x) > FLT_EPSILON) {
				return false;
			}
		}

		// ... [-extents.Y, extents.Y] and [Min(v0.Y,v1.Y,v2.Y), Max(v0.Y,v1.Y,v2.Y)] do not overlap
		if (Trimax(v0.y, v1.y, v2.y) < -extents.y || Trimin(v0.y, v1.y, v2.y) > extents.y) {
			if (Trimax(v0.y, v1.y, v2.y) - (-extents.y) > FLT_EPSILON ||
				Trimin(v0.y, v1.y, v2.y) - (extents.y) > FLT_EPSILON) {
				return false;
			}
		}

		// ... [-extents.Z, extents.Z] and [Min(v0.Z,v1.Z,v2.Z), Max(v0.Z,v1.Z,v2.Z)] do not overlap
		if (Trimax(v0.z, v1.z, v2.z) < -extents.z || Trimin(v0.z, v1.z, v2.z) > extents.z) {
			if (Trimax(v0.z, v1.z, v2.z) - (-extents.z) > FLT_EPSILON ||
				Trimin(v0.z, v1.z, v2.z) - (extents.z) > FLT_EPSILON) {
				return false;
			}
		}

		//// endregion

		//// region Test separating axis corresponding to triangle face normal (category 2)

		glm::vec3 plane_normal = { f0.y * f1.z - f0.z * f1.y, f0.z * f1.x - f0.x * f1.z, f0.x * f1.y - f0.y * f1.x };
		float plane_distance = std::fabs(plane_normal.x * v0.x + plane_normal.y * v0.y + plane_normal.z * v0.z);

		// Compute the projection interval radius of b onto L(t) = b.c + t * p.n
		r = extents.x * std::fabs(plane_normal.x) + extents.y * std::fabs(plane_normal.y) + extents.z * std::fabs(plane_normal.z);

		// Intersection occurs when plane distance falls within [-r,+r] interval
		if (plane_distance > r + FLT_EPSILON)
			return false;

		//// endregion

		return true;
	}

	ostream& operator<<(ostream& os, AABB const& aabb)
	{
		return os << "min : " << aabb.xyz << endl
			<< "max : " << aabb.XYZ << endl
			<< "center : " << aabb.GetCenter() << endl
			<< "x length : " << aabb.GetXLength() << endl
			<< "y length : " << aabb.GetYLength() << endl
			<< "z length : " << aabb.GetZLength() << endl;
	}

	namespace Intersection
	{
		bool Equals(const glm::vec3& a, const glm::vec3& b)
		{
			return 0.0001f > glm::distance(a, b);
		}

		bool PointInTriangle(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
		{
			glm::vec3 edge1 = v1 - v0;
			glm::vec3 edge2 = v2 - v0;
			glm::vec3 h = glm::cross(edge1, edge2);
			float a = glm::dot(h, edge2);

			if (a == 0.0f) {
				// The triangle is degenerate (a line or a point)
				return false;
			}

			glm::vec3 s = p - v0;
			float u = glm::dot(s, h) / a;
			if (u < 0.0f || u > 1.0f) {
				return false;
			}

			glm::vec3 q = glm::cross(s, edge1);
			float v = glm::dot(q, h) / a;
			if (v < 0.0f || (u + v) > 1.0f) {
				return false;
			}

			return true;
		}

		bool LinePlaneIntersection(const glm::vec3& l0, const glm::vec3& l1, const glm::vec3& pp, const glm::vec3& pn, glm::vec3& intersectionPoint) {
			auto lineDirection = l1 - l0;
			auto dotProduct = dot(lineDirection, pn);

			if (fabs(dotProduct) < 1e-6) {
				return false;
			}

			auto t = dot(pp - l0, pn) / dotProduct;

			if (t < 0) return false;
			else if (t > 1) return false;

			intersectionPoint = l0 + t * lineDirection;
			return true;
		}
	}

	NeonObject::NeonObject(const string& name)
		: name(name)
	{
	}

	NeonObject::~NeonObject()
	{
	}

	void NeonObject::OnKeyEvent(const KeyEvent& event)
	{
		for (auto& handler : keyEventHandlers)
		{
			handler(event);
		}
	}

	void NeonObject::OnMouseButtonEvent(const MouseButtonEvent& event)
	{
		for (auto& handler : mouseButtonEventHandlers)
		{
			handler(event);
		}
	}
	
	void NeonObject::OnCursorPosEvent(const CursorPosEvent& event)
	{
		for (auto& handler : cursorPosEventHandlers)
		{
			handler(event);
		}
	}
	
	void NeonObject::OnScrollEvent(const ScrollEvent& event)
	{
		for (auto& handler : scrollEventHandlers)
		{
			handler(event);
		}
	}

	void NeonObject::OnUpdate(double now, double timeDelta)
	{
		for (auto& handler : updateHandlers)
		{
			handler(now, timeDelta);
		}
	}







	time_point<high_resolution_clock> Time::Now()
	{
		return high_resolution_clock::now();
	}

	double Time::DeltaNano(const time_point<high_resolution_clock>& t)
	{
		return double(duration_cast<nanoseconds>(high_resolution_clock::now() - t).count());
	}

	double Time::DeltaNano(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1)
	{
		return double(duration_cast<nanoseconds>(t1 - t0).count());
	}

	double Time::DeltaMicro(const time_point<high_resolution_clock>& t)
	{
		return double(duration_cast<nanoseconds>(high_resolution_clock::now() - t).count()) / 1000.0;
	}

	double Time::DeltaMicro(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1)
	{
		return double(duration_cast<nanoseconds>(t1 - t0).count()) / 1000.0;
	}

	double Time::DeltaMili(const time_point<high_resolution_clock>& t)
	{
		return double(duration_cast<nanoseconds>(high_resolution_clock::now() - t).count()) / 1000000.0;
	}

	double Time::DeltaMili(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1)
	{
		return double(duration_cast<nanoseconds>(t1 - t0).count()) / 1000000.0;
	}

	Time::Time(const string& name)
		: name(name)
	{
		startedTime = Now();
		touchedTime = startedTime;
	}

	Time::~Time()
	{
		Stop();
	}

	void Time::Stop()
	{
		if (stoped == false)
		{
			if (name.empty() == false)
			{
				cout << "[" << name << "] ";
			}
			cout << DeltaMili(startedTime) << " miliseconds" << endl;
			stoped = true;
		}
	}

	void Time::Touch()
	{
		touchCount++;
		auto now = Now();

		if (name.empty() == false)
		{
			cout << "[" << name << " : " << touchCount << "] ";
		}
		else
		{
			cout << "[" << touchCount << "] ";
		}
		cout << DeltaMili(touchedTime, now) << " miliseconds" << endl;

		touchedTime = now;
	}

	int safe_stoi(const string& input)
	{
		if (input.empty())
		{
			return INT_MAX;
		}
		else
		{
			return stoi(input);
		}
	}

	float safe_stof(const string& input)
	{
		if (input.empty())
		{
			return FLT_MAX;
		}
		else
		{
			return stof(input);
		}
	}

	vector<string> split(const string& input, const string& delimiters, bool includeEmptyString)
	{
		vector<string> result;
		string piece;
		for (auto c : input)
		{
			bool contains = false;
			for (auto d : delimiters)
			{
				if (d == c)
				{
					contains = true;
					break;
				}
			}

			if (contains == false)
			{
				piece += c;
			}
			else
			{
				if (includeEmptyString || piece.empty() == false)
				{
					result.push_back(piece);
					piece.clear();
				}
			}
		}
		if (piece.empty() == false)
		{
			result.push_back(piece);
		}

		return result;
	}

	unsigned int NextPowerOf2(unsigned int n)
	{
		unsigned int p = 1;
		if (n && !(n & (n - 1)))
			return n;

		while (p < n)
			p <<= 1;

		return p;
	}

	VertexBufferObjectBase::VertexBufferObjectBase() {}
	VertexBufferObjectBase::~VertexBufferObjectBase() {}
}
