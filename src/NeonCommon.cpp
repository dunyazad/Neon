#include "NeonCommon.h"

namespace Neon
{
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

	void Time::Stop()
	{
		if (name.empty() == false)
		{
			cout << "[" << name << "] ";
		}
		cout << DeltaMili(startedTime) << " miliseconds" << endl;
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
}
