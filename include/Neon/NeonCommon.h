#pragma once

#pragma warning(disable: 4819)

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

#include <omp.h>

#include <Neon/TriangleTriangleOverlap.h>

#define FLT_VALID(x) ((x) < FLT_MAX / 2)
//#define FLT_VALID(x) ((x) < 3.402823466e+36F)
#define VECTOR3F_VALID(v) (FLT_VALID((v).x) && FLT_VALID((v).y) && FLT_VALID((v).z))
#define VECTOR3F_VALID_(v) (FLT_VALID((v).x()) && FLT_VALID((v).y()) && FLT_VALID((v).z()))
#define INT_VALID(x) ((x) != INT_MAX)
#define UINT_VALID(x) ((x) != UINT_MAX)
#define VECTOR3U_VALID(v) (UINT_VALID((v).x) && UINT_VALID((v).y) && UINT_VALID((v).z))
#define VECTOR3U_VALID_(v) (UINT_VALID((v).x()) && UINT_VALID((v).y()) && UINT_VALID((v).z()))


#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>

ostream& operator << (ostream& o, const glm::vec2& v);
ostream& operator << (ostream& o, const glm::vec3& v);
ostream& operator << (ostream& o, const glm::vec4& v);
ostream& operator << (ostream& o, const glm::mat3& m);
ostream& operator << (ostream& o, const glm::mat4& m);
ostream& operator << (ostream& o, const glm::quat& q);

bool operator < (const glm::vec3& a, const glm::vec3& b);
bool operator > (const glm::vec3& a, const glm::vec3& b);

namespace glm
{
	bool isClockwise(const glm::vec2 a, const glm::vec2 b, const glm::vec2 c);

	inline glm::vec4 white(1.0f, 1.0f, 1.0f, 1.0f);
	inline glm::vec4 black(0.0f, 0.0f, 0.0f, 1.0f);
	inline glm::vec4 darkgray(0.25f, 0.25f, 0.25f, 1.0f);
	inline glm::vec4 gray(0.5f, 0.5f, 0.5f, 1.0f);
	inline glm::vec4 lightgray(0.75f, 0.75f, 0.75f, 1.0f);
	inline glm::vec4 red(1.0f, 0.0f, 0.0f, 1.0f);
	inline glm::vec4 green(0.0f, 1.0f, 0.0f, 1.0f);
	inline glm::vec4 blue(0.0f, 0.0f, 1.0f, 1.0f);
	inline glm::vec4 yellow(1.0f, 1.0f, 0.0f, 1.0f);
	inline glm::vec4 cyan(0.0f, 1.0f, 1.0f, 1.0f);
	inline glm::vec4 magenta(1.0f, 0.0f, 1.0f, 1.0f);

	float distancePointToTriangle(const glm::vec3& point, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2);

	void to_json(json& j, const glm::vec3& p);
	void from_json(json& j, glm::vec3& p);

}

#define byte win_byte_override
#define NOMINMAX

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cudaGL.h>
#include <cuda_gl_interop.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

#define GLFW_DOUBLE_ACTION 4

void _CheckGLError(const char* file, int line);
#define CheckGLError() _CheckGLError(__FILE__, __LINE__)

#define SAFE_DELETE(x) if(x != nullptr) { delete x; x = nullptr; }

using namespace std::chrono;

float Trimax(float a, float b, float c);
float Trimin(float a, float b, float c);

#define Clamp(x, a, b) if((x) < (a)) (x) = (a); else if ((x) > (b)) (x) = (b);

namespace Neon
{
	extern json Settings;

#define PI 3.14159265359f

#define IntNAN std::numeric_limits<int>::quiet_NaN()
#define RealNAN std::numeric_limits<float>::quiet_NaN()
#define IntInfinity std::numeric_limits<int>::max()

#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

	template<typename T>
	T RandomInt(T from, T to)
	{
		std::random_device rd;
		std::mt19937 generator(rd());
		std::uniform_int_distribution<T> distribution(from, to);
		return distribution(generator);
	}

	template<typename T>
	T RandomReal(T from, T to, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr)
	{
		std::random_device rd;
		std::mt19937 generator(rd());
		std::uniform_real_distribution<T> distribution(from, to);
		return distribution(generator);
	}

	struct Ray
	{
		glm::vec3 origin = glm::zero<glm::vec3>();
		glm::vec3 direction = glm::zero<glm::vec3>();
	};

	struct Plane
	{
		glm::vec3 point = glm::zero<glm::vec3>();
		glm::vec3 normal = glm::zero<glm::vec3>();
	};

	struct AABB
	{
		glm::vec3 xyz = { FLT_MAX,  FLT_MAX,  FLT_MAX };
		glm::vec3 xyZ = { FLT_MAX,  FLT_MAX, -FLT_MAX };
		glm::vec3 xYz = { FLT_MAX, -FLT_MAX,  FLT_MAX };
		glm::vec3 xYZ = { FLT_MAX, -FLT_MAX, -FLT_MAX };
		glm::vec3 Xyz = { -FLT_MAX,  FLT_MAX,  FLT_MAX };
		glm::vec3 XyZ = { -FLT_MAX,  FLT_MAX, -FLT_MAX };
		glm::vec3 XYz = { -FLT_MAX, -FLT_MAX,  FLT_MAX };
		glm::vec3 XYZ = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
		glm::vec3 center = { 0.0,  0.0,  0.0 };
		glm::vec3 extents = { 0.0,  0.0,  0.0 };

		AABB(const glm::vec3& minPoint = { FLT_MAX, FLT_MAX, FLT_MAX }, const glm::vec3& maxPoint = { -FLT_MAX, -FLT_MAX, -FLT_MAX });

		void update();

		inline void Clear()
		{
			xyz = { FLT_MAX,  FLT_MAX,  FLT_MAX };
			xyZ = { FLT_MAX,  FLT_MAX, -FLT_MAX };
			xYz = { FLT_MAX, -FLT_MAX,  FLT_MAX };
			xYZ = { FLT_MAX, -FLT_MAX, -FLT_MAX };
			Xyz = { -FLT_MAX,  FLT_MAX,  FLT_MAX };
			XyZ = { -FLT_MAX,  FLT_MAX, -FLT_MAX };
			XYz = { -FLT_MAX, -FLT_MAX,  FLT_MAX };
			XYZ = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
			center = { 0.0,  0.0,  0.0 };
			extents = { 0.0,  0.0,  0.0 };
		}

		inline const glm::vec3& GetMinPoint() const { return xyz; }
		inline const glm::vec3& GetMaxPoint() const { return XYZ; }
		inline const glm::vec3& GetCenter() const { return center; }
		inline const glm::vec3& GetExtents() const { return extents; }
		inline float GetDiagonalLength() const { return glm::distance(XYZ, xyz); }

		inline void Expand(float x, float y, float z)
		{
			if (FLT_VALID(x) && x < xyz.x) { xyz.x = x; }
			if (FLT_VALID(y) && y < xyz.y) { xyz.y = y; }
			if (FLT_VALID(z) && z < xyz.z) { xyz.z = z; }

			if (FLT_VALID(x) && x > XYZ.x) { XYZ.x = x; }
			if (FLT_VALID(y) && y > XYZ.y) { XYZ.y = y; }
			if (FLT_VALID(z) && z > XYZ.z) { XYZ.z = z; }

			update();
		}

		inline void Expand(const glm::vec3& p)
		{
			if (FLT_VALID(p.x) && p.x < xyz.x) { xyz.x = p.x; }
			if (FLT_VALID(p.y) && p.y < xyz.y) { xyz.y = p.y; }
			if (FLT_VALID(p.z) && p.z < xyz.z) { xyz.z = p.z; }

			if (FLT_VALID(p.x) && p.x > XYZ.x) { XYZ.x = p.x; }
			if (FLT_VALID(p.y) && p.y > XYZ.y) { XYZ.y = p.y; }
			if (FLT_VALID(p.z) && p.z > XYZ.z) { XYZ.z = p.z; }

			update();
		}

		inline float GetXLength() const { return XYZ.x - xyz.x; }
		inline float GetYLength() const { return XYZ.y - xyz.y; }
		inline float GetZLength() const { return XYZ.z - xyz.z; }

		inline bool Contains(const glm::vec3& p) const
		{
			return (xyz.x <= p.x && p.x <= XYZ.x) &&
				(xyz.y <= p.y && p.y <= XYZ.y) &&
				(xyz.z <= p.z && p.z <= XYZ.z);
		}

		inline bool Intersects(const AABB& other) const
		{
			if ((xyz.x > other.XYZ.x) && (xyz.y > other.XYZ.y) && (xyz.z > other.XYZ.z)) { return false; }
			if ((XYZ.x < other.xyz.x) && (XYZ.y < other.xyz.y) && (XYZ.z < other.xyz.z)) { return false; }

			if (Contains(other.xyz)) return true;
			if (Contains(other.xyZ)) return true;
			if (Contains(other.xYz)) return true;
			if (Contains(other.xYZ)) return true;
			if (Contains(other.Xyz)) return true;
			if (Contains(other.XyZ)) return true;
			if (Contains(other.XYz)) return true;
			if (Contains(other.XYZ)) return true;

			if (other.Contains(xyz)) return true;
			if (other.Contains(xyZ)) return true;
			if (other.Contains(xYz)) return true;
			if (other.Contains(xYZ)) return true;
			if (other.Contains(Xyz)) return true;
			if (other.Contains(XyZ)) return true;
			if (other.Contains(XYz)) return true;
			if (other.Contains(XYZ)) return true;

			return false;
		}

		bool IntersectsRay(const Ray& ray, vector<glm::vec3>& intersections);

		bool IntersectsTriangle(const glm::vec3& tp0, const glm::vec3& tp1, const glm::vec3& tp2);
	};

	ostream& operator<<(ostream& os, const AABB& aabb);

	namespace Intersection
	{
		bool Equals(const glm::vec3& a, const glm::vec3& b);
		bool PointInTriangle(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2);
		bool LinePlaneIntersection(const glm::vec3& l0, const glm::vec3& l1, const glm::vec3& pn, const glm::vec3& pp, glm::vec3& intersectionPoint);
		bool TriangleTriangleIntersection(
			const glm::vec3& a0, const glm::vec3& a1, const glm::vec3& a2,
			const glm::vec3& b0, const glm::vec3& b1, const glm::vec3& b2,
			glm::vec3& i0, glm::vec3& i1);
	}

	typedef unsigned long ID;

	class NeonObject;

	struct KeyEvent
	{
		NeonObject* target = nullptr;
		GLFWwindow* window = nullptr;
		int key = GLFW_KEY_UNKNOWN;
		int scancode = -1;
		int action = -1;
		int mods = -1;
	};

	struct MouseButtonEvent
	{
		NeonObject* target = nullptr;
		GLFWwindow* window = nullptr;
		int button = -1;
		int action = -1;
		int mods = -1;
		double xpos = 0.0;
		double ypos = 0.0;
	};

	struct CursorPosEvent
	{
		NeonObject* target = nullptr;
		GLFWwindow* window = nullptr;
		double xpos = 0.0;
		double ypos = 0.0;
	};

	struct ScrollEvent
	{
		NeonObject* target = nullptr;
		GLFWwindow* window = nullptr;
		double xoffset = 0;
		double yoffset = 0;
	};

	class NeonObject
	{
	public:
		NeonObject(const string& name);
		virtual ~NeonObject();

		inline const string& GetName() const { return name; }

		virtual void OnKeyEvent(const KeyEvent& event);
		virtual void OnMouseButtonEvent(const MouseButtonEvent& event);
		virtual void OnCursorPosEvent(const CursorPosEvent& event);
		virtual void OnScrollEvent(const ScrollEvent& event);

		inline void AddKeyEventHandler(function<void(const KeyEvent&)> handler) { keyEventHandlers.push_back(handler); }
		inline void AddMouseButtonEventHandler(function<void(const MouseButtonEvent&)> handler) { mouseButtonEventHandlers.push_back(handler); }
		inline void AddCursorPosEventHandler(function<void(const CursorPosEvent&)> handler) { cursorPosEventHandlers.push_back(handler); }
		inline void AddScrollEventHandler(function<void(const ScrollEvent&)> handler) { scrollEventHandlers.push_back(handler); }

		virtual void OnUpdate(double now, double timeDelta);

		inline void AddUpdateHandler(function<void(double, double)> handler) { updateHandlers.push_back(handler); }
		//inline void RemoveUpdateHandler(function<void(double, double)> handler) { updateHandlers.erase(handler); }

	protected:
		string name;

		vector<function<void(const KeyEvent&)>> keyEventHandlers;
		vector<function<void(const MouseButtonEvent&)>> mouseButtonEventHandlers;
		vector<function<void(const CursorPosEvent&)>> cursorPosEventHandlers;
		vector<function<void(const ScrollEvent&)>> scrollEventHandlers;

		vector<function<void(double, double)>> updateHandlers;
	};

	class Time
	{
	public:
		static time_point<high_resolution_clock> Now();
		static double DeltaNano(const time_point<high_resolution_clock>& t);
		static double DeltaNano(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1);

		static double DeltaMicro(const time_point<high_resolution_clock>& t);
		static double DeltaMicro(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1);

		static double DeltaMili(const time_point<high_resolution_clock>& t);
		static double DeltaMili(const time_point<high_resolution_clock>& t0, const time_point<high_resolution_clock>& t1);


		Time(const string& name);
		~Time();
		void Touch();
		void Stop();

	protected:
		bool stoped = false;
		string name;
		time_point<high_resolution_clock> startedTime;
		int touchCount = 0;
		time_point<high_resolution_clock> touchedTime;
	};

	unsigned int NextPowerOf2(unsigned int n);

	class VertexBufferObjectBase
	{
	public:
		VertexBufferObjectBase();
		~VertexBufferObjectBase();

		enum BufferType { VERTEX_BUFFER, NORMAL_BUFFER, COLOR_BUFFER, UV_BUFFER, INDEX_BUFFER };

		virtual void Bind() = 0;
		virtual void Unbind() = 0;
		virtual void Upload() = 0;

		virtual void Clear() = 0;
	};

	int safe_stoi(const string& input);
	float safe_stof(const string& input);
	
	vector<string> split(const string& input, const string& delimiters, bool includeEmptyString = false);
}