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

	void Entity::OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		NeonObject::OnKeyEvent(window, key, scancode, action, mods);

		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				component->OnKeyEvent(window, key, scancode, action, mods);
			}
		}
	}

	void Entity::OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods)
	{
		NeonObject::OnMouseButtonEvent(window, button, action, mods);

		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				component->OnMouseButtonEvent(window, button, action, mods);
			}
		}
	}

	void Entity::OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos)
	{
		NeonObject::OnCursorPosEvent(window, xpos, ypos);

		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				component->OnCursorPosEvent(window, xpos, ypos);
			}
		}
	}

	void Entity::OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset)
	{
		NeonObject::OnScrollEvent(window, xoffset, yoffset);

		for (auto& kvp : components)
		{
			for (auto& component : kvp.second)
			{
				component->OnScrollEvent(window, xoffset, yoffset);
			}
		}
	}
}
