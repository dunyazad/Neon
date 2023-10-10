#include <Neon/NeonSystem.h>
#include <Neon/Neon.h>

namespace Neon
{
	SystemBase::SystemBase(Application* application)
		: application(application)
	{
	}

	SystemBase::~SystemBase()
	{
	}

	RenderSystem::RenderSystem(Application* application)
		: SystemBase(application)
	{

	}

	RenderSystem::~RenderSystem()
	{

	}

	void RenderSystem::Frame(float timeDelta)
	{
		auto entities = application->GetEntities();
		for (auto& kvp : entities)
		{
			auto components = kvp.second->GetComponents<RenderData>();
			for (auto& component : components)
			{
				auto renderData = (RenderData*)component;
				(*renderData->GetShaders().begin())->Use();
				renderData->Bind();
				glDrawElements(GL_TRIANGLES, (GLsizei)renderData->GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);
			}
		}
	}
}
