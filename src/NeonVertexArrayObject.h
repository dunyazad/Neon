#pragma once

#include "NeonCommon.h"

namespace Neon
{
	class VertexArrayObject
	{
	public:
		VertexArrayObject();
		~VertexArrayObject();

		inline unsigned int ID() { return id; }

		virtual void Initialize();
		virtual void Terminate();
		void Bind();
		void Unbind();

	protected:
		unsigned int id = -1;
	};
}