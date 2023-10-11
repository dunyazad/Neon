#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class ComponentBase
	{
	public:
		ComponentBase(const string& name);
		virtual ~ComponentBase();

		virtual void OnUpdate(float now, float timeDelta);

		inline void AddUpdateCallback(function<void(float, float)> callback) { updateCallbacks.push_back(callback); }

	protected:
		string name;

		vector<function<void(float, float)>> updateCallbacks;
	};
}
