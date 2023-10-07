#pragma once

#include <Neon/NeonCommon.h>

namespace Neon
{
	class VertexArrayObject
	{
	public:
		VertexArrayObject();
		~VertexArrayObject();

		inline unsigned int ID() { return id; }

		void Bind();
		void Unbind();

	protected:
		unsigned int id = -1;
	};
}