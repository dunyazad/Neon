#include <Neon/NeonScene.h>
#include <Neon/NeonEntity.h>

namespace Neon
{
	Scene::Scene(const string& name)
		: name(name), transformUpdateSystem(this), renderSystem(this)
	{
	}

	Scene::~Scene()
	{
		for (auto& kvp : entities)
		{
			SAFE_DELETE(kvp.second);
		}

		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				SAFE_DELETE(component);
			}
		}
	}

	void Scene::Frame(float now, float timeDelta)
	{
		transformUpdateSystem.Frame((float)now, (float)timeDelta);
		renderSystem.Frame((float)now, (float)timeDelta);
	}

	Entity* Scene::GetEntity(const string& name)
	{
		if (0 != entities.count(name))
		{
			return entities[name];
		}
		else
		{
			return nullptr;
		}
	}

	Entity* Scene::CreateEntity(const string& name)
	{
		if (0 != entities.count(name))
		{
			return nullptr;
		}
		else
		{
			auto entity = new Entity();
			entities[name] = entity;
			return entity;
		}
	}
}
