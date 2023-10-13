#include <Neon/NeonEntity.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	Entity::Entity(const string& name, Scene* scene)
		: NeonObject(name), scene(scene)
	{
	}

	Entity::~Entity()
	{
	}

	void Entity::OnKeyEvent(const KeyEvent& event)
	{
		NeonObject::OnKeyEvent(event);

		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				component->OnKeyEvent(event);
			}
		}
	}

	void Entity::OnMouseButtonEvent(const MouseButtonEvent& event)
	{
		NeonObject::OnMouseButtonEvent(event);

		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				component->OnMouseButtonEvent(event);
			}
		}
	}

	void Entity::OnCursorPosEvent(const CursorPosEvent& event)
	{
		NeonObject::OnCursorPosEvent(event);

		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				component->OnCursorPosEvent(event);
			}
		}
	}

	void Entity::OnScrollEvent(const ScrollEvent& event)
	{
		NeonObject::OnScrollEvent(event);

		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				component->OnScrollEvent(event);
			}
		}
	}
}
