#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class ComponentBase;

	class Entity
	{
	public:
		Entity(const string& name);
		~Entity();

		template<class T>
		void AddComponent(T* component)
		{
			if (nullptr != component)
			{
				components[&typeid(T)].push_back(component);
			}
		}

		template<class T>
		vector<ComponentBase*> GetComponents()
		{
			return components[&typeid(T)];
		}

		template<class T>
		T* GetComponent(int index)
		{
			auto cv = components[&typeid(T)];
			if (0 == cv.size()) return nullptr;
			else if (index > cv.size() - 1) return nullptr;
			else return (T*)cv[index];
		}

		inline const string& GetName() const { return name; }

	protected:
		string name;

		map<const type_info*, vector<ComponentBase*>> components;
	};
}
