#include <Neon/Component/NeonEventSubscriber.h>

#include <Neon/NeonEntity.h>

namespace Neon
{
	EventSubscriber::EventSubscriber(const string& name, Entity* entity)
		: ComponentBase(name), entity(entity)
	{
	}

	EventSubscriber::~EventSubscriber()
	{
	}

	void EventSubscriber::OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		//cout << "Entity : " << entity->GetName() << endl;
		//cout << "Window : " << window << endl;
		//cout << "key : " << key << endl;

		if (keyEventCallback)
		{
			keyEventCallback(window, key, scancode, action, mods);
		}
	}

	void EventSubscriber::OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods)
	{
		//cout << "Entity : " << entity->GetName() << endl;
		//cout << "Window : " << window << endl;
		//cout << "button : " << button << endl;

		if (mouseButtonEventCallback)
		{
			mouseButtonEventCallback(window, button, action, mods);
		}
	}

	void EventSubscriber::OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos)
	{
		//cout << "Entity : " << entity->GetName() << endl;
		//cout << "Window : " << window << endl;
		//cout << "x : " << xpos << " , y : " << ypos << endl;

		if (cursorPosEventCallback)
		{
			cursorPosEventCallback(window, xpos, ypos);
		}
	}

	void EventSubscriber::OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset)
	{
		//cout << "Entity : " << entity->GetName() << endl;
		//cout << "Window : " << window << endl;
		//cout << "xoffset : " << xoffset << " , yoffset : " << yoffset << endl;

		if (scrollEventCallback)
		{
			scrollEventCallback(window, xoffset, yoffset);
		}
	}
}
