#pragma once

#pragma warning(disable : 4819)

#include <algorithm>
#include <chrono>
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
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>
using namespace std;

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

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#define GLFW_DOUBLE_ACTION 4

void _CheckGLError(const char* file, int line);
#define CheckGLError() _CheckGLError(__FILE__, __LINE__)

#define SAFE_DELETE(x) if(x != nullptr) { delete x; x = nullptr; }

using namespace std::chrono;

namespace Neon
{
	extern json Settings;

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

		virtual void OnUpdate(float now, float timeDelta);

		inline void AddUpdateHandler(function<void(float, float)> handler) { updateHandlers.push_back(handler); }
		//inline void RemoveUpdateHandler(function<void(float, float)> handler) { updateHandlers.erase(handler); }

	protected:
		string name;

		vector<function<void(const KeyEvent&)>> keyEventHandlers;
		vector<function<void(const MouseButtonEvent&)>> mouseButtonEventHandlers;
		vector<function<void(const CursorPosEvent&)>> cursorPosEventHandlers;
		vector<function<void(const ScrollEvent&)>> scrollEventHandlers;

		vector<function<void(float, float)>> updateHandlers;
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