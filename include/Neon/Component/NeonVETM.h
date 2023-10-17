#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/Component/NeonComponent.h>

namespace Neon
{
	class Mesh;

	class VETM : public ComponentBase
	{
	public:
		VETM(const string& name, Mesh* mesh);
		~VETM();

	private:
		Mesh* mesh = nullptr;
	};
}