#include "MiniMath.h"
#include <sstream>
#include <algorithm>

namespace MiniMath
{
	float clamp(float f, float minf, float maxf)
	{
		return min(max(f, minf), maxf);
	}

	V3::V3()
		: x(0.0f), y(0.0f), z(0.0f) {}
	V3::V3(int ix, int iy, int iz)
		:x(ix), y(iy), z(iz) {}
	V3::V3(double dx, double dy, double dz)
		: x(dx), y(dy), z(dz) {}
	V3::V3(const char* c)
	{
		string code(c);
		std::istringstream iss(code);
		float value;
		iss >> std::noskipws >> value;

		if (iss.eof() && !iss.fail())
		{
			x = y = value;
		}
		else
		{
			std::transform(code.begin(), code.end(), code.begin(), [](unsigned char c) { return std::tolower(c); });

			if (code == "nan")
			{
				x = y = std::numeric_limits<float>::quiet_NaN();
			}
			else if (code == "zero")
			{
				x = y = 0.0;
			}
			else if (code == "one")
			{
				x = y = 1.0;
			}
			else if (code == "half")
			{
				x = y = 0.5;
			}
			else if (code == "red")
			{
				x = 1.0f;
			}
			else if (code == "green")
			{
				y = 1.0f;
			}
			else if (code == "blue")
			{
				z = 1.0f;
			}
			else if (code == "black")
			{
				x = y = z = 0.0f;
			}
			else if (code == "gray")
			{
				x = y = z = 0.5f;
			}
			else if (code == "white")
			{
				x = y = z = 1.0f;
			}
		}
	}


	const V3& V3::operator += (float scalar) { x += scalar; y += scalar; z += scalar; return *this; }
	const V3& V3::operator -= (float scalar) { x -= scalar; y -= scalar; z -= scalar; return *this; }
	const V3& V3::operator *= (float scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
	const V3& V3::operator /= (float scalar) { x /= scalar; y /= scalar; z /= scalar; return *this; }

	const V3& V3::operator += (const V3& other) { x += other.x; y += other.y; z += other.z; return *this; }
	const V3& V3::operator -= (const V3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }

	float V3::operator[](int index) { return *(&this->x + index); }

	V3 operator - (const V3& v) { return { -v.x, -v.y, -v.z }; }
	V3 operator + (const V3& v, float scalar) { return { v.x + scalar, v.y + scalar, v.z + scalar }; }
	V3 operator + (float scalar, const V3& v) { return { v.x + scalar, v.y + scalar, v.z + scalar }; }
	V3 operator - (const V3& v, float scalar) { return { v.x - scalar, v.y - scalar, v.z - scalar }; }
	V3 operator - (float scalar, const V3& v) { return { v.x - scalar, v.y - scalar, v.z - scalar }; }
	V3 operator * (const V3& v, float scalar) { return { v.x * scalar, v.y * scalar, v.z * scalar }; }
	V3 operator * (float scalar, const V3& v) { return { v.x * scalar, v.y * scalar, v.z * scalar }; }
	V3 operator / (const V3& v, float scalar) { return { v.x / scalar, v.y / scalar, v.z / scalar }; }
	V3 operator / (float scalar, const V3& v) { return { v.x / scalar, v.y / scalar, v.z / scalar }; }

	V3 operator + (const V3& a, const V3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
	V3 operator - (const V3& a, const V3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }

	float magnitude(const V3& v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
	V3 normalize(const V3& v)
	{
		float mag = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
		if (mag != 0.0f)
		{
			return { v.x / mag, v.y / mag, v.z / mag };
		}
		return v;
	}

	float dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
	V3 cross(const V3& a, const V3& b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

	float distance(const V3& a, const V3& b)
	{
		return sqrtf((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z));
	}

	float angle(const V3& a, const V3& b)
	{
		return acosf(clamp(dot(a, b), -1.0f, 1.0f));
	}

	V3 calculateMean(const std::vector<V3>& vectors)
	{
		V3 mean;
		for (const V3& vec : vectors)
		{
			mean.x += vec.x;
			mean.y += vec.y;
			mean.z += vec.z;
		}
		mean.x /= vectors.size();
		mean.y /= vectors.size();
		mean.z /= vectors.size();
		return mean;
	}

	void centerData(std::vector<V3>& vectors, const V3& mean)
	{
		for (V3& vec : vectors)
		{
			vec.x -= mean.x;
			vec.y -= mean.y;
			vec.z -= mean.z;
		}
	}

	void calculateCovarianceMatrix(const std::vector<V3>& vectors, V3& eigenvalues, V3& eigenvector1, V3& eigenvector2, V3& eigenvector3)
	{
		int n = vectors.size();

		for (const V3& vec : vectors)
		{
			eigenvalues.x += vec.x * vec.x;
			eigenvalues.y += vec.y * vec.y;
			eigenvalues.z += vec.z * vec.z;

			eigenvector1.x += vec.x * vec.x;
			eigenvector2.x += vec.x * vec.y;
			eigenvector3.x += vec.x * vec.z;

			eigenvector2.y += vec.y * vec.y;
			eigenvector3.y += vec.y * vec.z;

			eigenvector3.z += vec.z * vec.z;
		}

		eigenvalues.x /= n;
		eigenvalues.y /= n;
		eigenvalues.z /= n;

		eigenvector1.x /= n;
		eigenvector2.x /= n;
		eigenvector3.x /= n;

		eigenvector2.y /= n;
		eigenvector3.y /= n;

		eigenvector3.z /= n;
	}

	Quaternion::Quaternion(float scalar, float i, float j, float k)
		: w(scalar), x(i), y(j), z(k) {}

	Quaternion::Quaternion(float radian, const V3& axis)
	{
		float halfAngle = radian * 0.5f;
		float sinHalfAngle = sinf(halfAngle);
		w = cosf(halfAngle);
		x = axis.x * sinHalfAngle;
		y = axis.y * sinHalfAngle;
		z = axis.z * sinHalfAngle;
	}

	Quaternion conjugate(const Quaternion q) { return { q.w, -q.x, -q.y, -q.z }; }

	Quaternion operator*(const Quaternion& q1, const Quaternion& q2)
	{
		float w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
		float x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
		float y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
		float z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;

		return Quaternion(w, x, y, z);
	}

	V3 rotate(const V3& v, const Quaternion& q)
	{
		auto qv = V3{ q.x, q.y, q.z };
		auto uv = cross(qv, v);
		auto uuv = cross(qv, uv);

		return v + ((uv * q.w) + uuv) * 2.0f;
	}
}
