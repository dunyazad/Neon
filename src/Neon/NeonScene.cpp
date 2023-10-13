#include <Neon/NeonScene.h>
#include <Neon/NeonEntity.h>
#include <Neon/NeonDebugEntity.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	Scene::Scene(const string& name, Window* window)
		: name(name), window(window), transformUpdateSystem(this), renderSystem(this), eventSystem(this)
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
			auto entity = new Entity(name, this);
			entities[name] = entity;
			return entity;
		}
	}

	DebugEntity* Scene::GetDebugEntity(const string& name)
	{
		if (0 != debugEntities.count(name))
		{
			return debugEntities[name];
		}
		else
		{
			return nullptr;
		}
	}

	DebugEntity* Scene::CreateDebugEntity(const string& name)
	{
		if (0 != debugEntities.count(name))
		{
			return nullptr;
		}
		else
		{
			auto entity = new DebugEntity(name, this);
			debugEntities[name] = entity;
			return entity;
		}
	}
}
