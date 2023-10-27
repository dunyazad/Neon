#include <Neon/System/NeonEntityUpdateSystem.h>

#include <Neon/NeonDebugEntity.h>
#include <Neon/NeonEntity.h>
#include <Neon/NeonScene.h>

#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	EntityUpdateSystem::EntityUpdateSystem(Scene* scene)
		: SystemBase(scene)
	{
	}

	EntityUpdateSystem::~EntityUpdateSystem()
	{
	}

	void EntityUpdateSystem::Frame(double now, double timeDelta)
	{
		{
			auto entities = scene->GetEntities();
			for (auto& entity : entities)
			{
				entity.second->OnUpdate(now, timeDelta);
			}
		}
		{
			auto entities = scene->GetDebugEntities();
			for (auto& entity : entities)
			{
				entity->OnUpdate(now, timeDelta);
			}
		}

		auto components = scene->GetAllComponents();
		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				component->OnUpdate(now, timeDelta);
			}
		}
	}
}
