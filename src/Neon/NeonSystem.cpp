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
		auto components = application->GetComponents<RenderData>();

		for (auto& c : components)
		{
			auto component = (RenderData*)c;
			(*component->GetShaders().begin())->use();

			component->Bind();
			glDrawElements(GL_TRIANGLES, (GLsizei)component->GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);
		}
	}
}
