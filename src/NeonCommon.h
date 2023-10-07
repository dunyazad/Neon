#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
#include <vector>
using namespace std;

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#define SAFE_DELETE(x) if(x != nullptr) { delete x; x = nullptr; }

using namespace std::chrono;

namespace Neon
{
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
		void Touch();
		void Stop();

	protected:
		string name;
		time_point<high_resolution_clock> startedTime;
		int touchCount = 0;
		time_point<high_resolution_clock> touchedTime;
	};
}