#include <Neon/NeonCommon.h>

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
	NeonObject::NeonObject(const string& name)
		: name(name)
	{
	}

	NeonObject::~NeonObject()
	{
	}

	void NeonObject::OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		if (keyEventCallback) keyEventCallback(window, key, scancode, action, mods);
	}

	void NeonObject::OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods)
	{
		if (mouseButtonEventCallback) mouseButtonEventCallback(window, button, action, mods);
	}
	
	void NeonObject::OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos)
	{
		if (cursorPosEventCallback) cursorPosEventCallback(window, xpos, ypos);
	}
	
	void NeonObject::OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset)
	{
		if (scrollEventCallback) scrollEventCallback(window, xoffset, yoffset);
	}

	void NeonObject::OnUpdate(float now, float timeDelta)
	{
		if (updateCallback) updateCallback(now, timeDelta);
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
