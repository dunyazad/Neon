#include <Neon/NeonCommon.h>

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

	namespace Intersection
	{
		bool Equals(const glm::vec3& a, const glm::vec3& b)
		{
			return FLT_EPSILON > glm::distance(a, b);
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
