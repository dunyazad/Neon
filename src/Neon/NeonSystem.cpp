#include <Neon/NeonSystem.h>
#include <Neon/NeonEntity.h>
#include <Neon/NeonScene.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/NeonWindow.h>
#include <Neon/Component/NeonCamera.h>
#include <Neon/Component/NeonEventSubscriber.h>
#include <Neon/Component/NeonLight.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonShader.h>
#include <Neon/Component/NeonTexture.h>
#include <Neon/Component/NeonTransform.h>

namespace Neon
{
	SystemBase::SystemBase(Scene* scene)
		: scene(scene)
	{
	}

	SystemBase::~SystemBase()
	{
	}

	TransformUpdateSystem::TransformUpdateSystem(Scene* scene)
		: SystemBase(scene)
	{
	}

	TransformUpdateSystem::~TransformUpdateSystem()
	{
	}

	void TransformUpdateSystem::Frame(float now, float timeDelta)
	{
		auto components = scene->GetComponents<Transform>();
		for (auto& component : components)
		{
			((Transform*)component)->OnUpdate(now, timeDelta);
		}
	}


	RenderSystem::RenderSystem(Scene* scene)
		: SystemBase(scene)
	{

	}

	RenderSystem::~RenderSystem()
	{

	}

	void RenderSystem::Frame(float now, float timeDelta)
	{
		auto camera = scene->GetMainCamera();
		camera->OnUpdate(now, timeDelta);

		auto light = scene->GetMainLight();
		light->OnUpdate(now, timeDelta);

		auto entities = scene->GetEntities();
		for (auto& kvp : entities)
		{
			auto entity = kvp.second;

			auto shader = entity->GetComponent<Shader>(0);
			if (nullptr != shader)
			{
				shader->Use();
			}

			if (nullptr != camera)
			{
				if (nullptr != shader)
				{
					shader->SetUniformFloat4x4("projection", glm::value_ptr(camera->projectionMatrix));
					shader->SetUniformFloat4x4("view", glm::value_ptr(camera->viewMatrix));
				}
			}

			if (nullptr != light)
			{
				if (nullptr != shader)
				{
					if (nullptr != camera)
					{
						shader->SetUniformFloat3("cameraPosition", glm::value_ptr(camera->position));
					}
					//auto lightDir = glm::vec3(0.0f, 0.0f, 10.0f);
					shader->SetUniformFloat3("lightPosition", glm::value_ptr(light->position));
					//shader->SetUniformFloat3("lightPos", glm::value_ptr(lightPos));
					shader->SetUniformFloat3("lightDirection", glm::value_ptr(light->direction));
					//shader->SetUniformFloat3("lightDirection", glm::value_ptr(lightDir));
					shader->SetUniformFloat3("lightColor", glm::value_ptr(light->color));
				}
			}

			auto transform = entity->GetComponent<Transform>(0);
			if (nullptr != transform)
			{
				if (nullptr != shader)
				{
					shader->SetUniformFloat4x4("model", glm::value_ptr(transform->absoluteTransform));
				}
			}
			else
			{
				glm::mat4 identity = glm::identity<glm::mat4>();

				if (nullptr != shader)
				{
					shader->SetUniformFloat4x4("model", glm::value_ptr(identity));
				}
			}

			auto texture = entity->GetComponent<Texture>(0);
			if (nullptr != texture)
			{
				texture->Bind();
			}

			auto mesh = entity->GetComponent<Mesh>(0);
			if (nullptr != mesh)
			{
				mesh->Bind();
				glDrawElements(mesh->GetDrawingMode(), (GLsizei)mesh->GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);
			}
		}
	}

	set<EventSystem*> EventSystem::instances;

	EventSystem::EventSystem(Scene* scene)
		: SystemBase(scene)
	{
		if (0 == instances.size())
		{
			//struct {
			//	GLFWwindowposfun          pos;
			//	GLFWwindowsizefun         size;
			//	GLFWwindowclosefun        close;
			//	GLFWwindowrefreshfun      refresh;
			//	GLFWwindowfocusfun        focus;
			//	GLFWwindowiconifyfun      iconify;
			//	GLFWwindowmaximizefun     maximize;
			//	GLFWframebuffersizefun    fbsize;
			//	GLFWwindowcontentscalefun scale;
			//	GLFWmousebuttonfun        mouseButton;
			//	GLFWcursorposfun          cursorPos;
			//	GLFWcursorenterfun        cursorEnter;
			//	GLFWscrollfun             scroll;
			//	GLFWkeyfun                key;
			//	GLFWcharfun               character;
			//	GLFWcharmodsfun           charmods;
			//	GLFWdropfun               drop;
			//} callbacks;

			glfwSetKeyCallback(scene->GetWindow()->GetGLFWWindow(), EventSystem::KeyCallback);
			glfwSetMouseButtonCallback(scene->GetWindow()->GetGLFWWindow(), EventSystem::MouseButtonCallback);
			glfwSetCursorPosCallback(scene->GetWindow()->GetGLFWWindow(), EventSystem::CursorPosCallback);
			glfwSetScrollCallback(scene->GetWindow()->GetGLFWWindow(), EventSystem::ScrollCallback);
		}

		instances.insert(this);
	}

	EventSystem::~EventSystem()
	{
	}

	void EventSystem::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		for (auto& instance : instances)
		{
			instance->OnKeyEvent(window, key, scancode, action, mods);
		}
	}

	void EventSystem::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		for (auto& instance : instances)
		{
			instance->OnMouseButtonEvent(window, button, action, mods);
		}
	}

	void EventSystem::CursorPosCallback(GLFWwindow* window, double xpos, double ypos)
	{
		for (auto& instance : instances)
		{
			instance->OnCursorPosEvent(window, xpos, ypos);
		}
	}

	void EventSystem::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		for (auto& instance : instances)
		{
			instance->OnScrollEvent(window, xoffset, yoffset);
		}
	}

	void EventSystem::OnKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		auto components = scene->GetComponents<EventSubscriber>();
		for (auto& component : components)
		{
			auto subscriber = (EventSubscriber*)component;
			subscriber->OnKeyEvent(window, key, scancode, action, mods);
		}
	}

	void EventSystem::OnMouseButtonEvent(GLFWwindow* window, int button, int action, int mods)
	{
		auto components = scene->GetComponents<EventSubscriber>();
		for (auto& component : components)
		{
			auto subscriber = (EventSubscriber*)component;
			subscriber->OnMouseButtonEvent(window, button, action, mods);
		}
	}

	void EventSystem::OnCursorPosEvent(GLFWwindow* window, double xpos, double ypos)
	{
		auto components = scene->GetComponents<EventSubscriber>();
		for (auto& component : components)
		{
			auto subscriber = (EventSubscriber*)component;
			subscriber->OnCursorPosEvent(window, xpos, ypos);
		}
	}

	void EventSystem::OnScrollEvent(GLFWwindow* window, double xoffset, double yoffset)
	{
		auto components = scene->GetComponents<EventSubscriber>();
		for (auto& component : components)
		{
			auto subscriber = (EventSubscriber*)component;
			subscriber->OnScrollEvent(window, xoffset, yoffset);
		}
	}
}
