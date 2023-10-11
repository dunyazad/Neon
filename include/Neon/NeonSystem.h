#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class Scene;
	class Entity;

	class SystemBase
	{
	public:
		SystemBase(Scene* scene);
		~SystemBase();

	protected:
		Scene* scene;
	};

	class TransformUpdateSystem : public SystemBase
	{
	public:
		TransformUpdateSystem(Scene* scene);
		~TransformUpdateSystem();

		void Frame(float now, float timeDelta);

	private:

	};

	class RenderSystem : public SystemBase
	{
	public:
		RenderSystem(Scene* scene);
		~RenderSystem();

		void Frame(float now, float timeDelta);

	private:

	};

	class EventSystem : public SystemBase
	{
	public:
		EventSystem(Scene* scene);
		~EventSystem();
		
		static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
		static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
		static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

		void OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods);
		void OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods);
		void OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos);
		void OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset);

		inline void SubscribeKeyEvent(Entity* entity) { keyEventSubscribers.insert(entity); }
		inline void UnsubscribeKeyEvent(Entity* entity) { keyEventSubscribers.erase(entity); }

		inline void SubscribeMouseButtonEvent(Entity* entity) { mouseButtonEventSubscribers.insert(entity); }
		inline void UnsubscribeMouseButtonEvent(Entity* entity) { mouseButtonEventSubscribers.erase(entity); }

		inline void SubscribeCursorPosEvent(Entity* entity) { cursorPosEventSubscribers.insert(entity); }
		inline void UnsubscribeCursorPosEvent(Entity* entity) { cursorPosEventSubscribers.erase(entity); }

		inline void SubscribeScrollEvent(Entity* entity) { scrollEventSubscribers.insert(entity); }
		inline void UnsubscribeScrollEvent(Entity* entity) { scrollEventSubscribers.erase(entity); }

	protected:
		static set<EventSystem*> instances;

		set<Entity*> keyEventSubscribers;
		set<Entity*> mouseButtonEventSubscribers;
		set<Entity*> cursorPosEventSubscribers;
		set<Entity*> scrollEventSubscribers;
	};
}
