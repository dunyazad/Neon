#include <Neon/NeonEntity.h>

namespace Neon
{
	Entity::Entity(const string& name)
		: name(name)
	{
	}

	Entity::~Entity()
	{
	}

	void Entity::OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		cout << "Entity : " << name << endl;
		cout << "Window : " << window << endl;
		cout << "key : " << key << endl;

		if (keyEventCallback)
		{
			keyEventCallback(window, key, scancode, action, mods);
		}
	}

	void Entity::OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods)
	{
		cout << "Entity : " << name << endl;
		cout << "Window : " << window << endl;
		cout << "button : " << button << endl;

		if (mouseButtonEventCallback)
		{
			mouseButtonEventCallback(window, button, action, mods);
		}
	}

	void Entity::OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos)
	{
		cout << "Entity : " << name << endl;
		cout << "Window : " << window << endl;
		cout << "x : " << xpos << " , y : " << ypos << endl;

		if (cursorPosEventCallback)
		{
			cursorPosEventCallback(window, xpos, ypos);
		}
	}

	void Entity::OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset)
	{
		cout << "Entity : " << name << endl;
		cout << "Window : " << window << endl;
		cout << "xoffset : " << xoffset << " , yoffset : " << yoffset << endl;

		if (scrollEventCallback)
		{
			scrollEventCallback(window, xoffset, yoffset);
		}
	}
}
