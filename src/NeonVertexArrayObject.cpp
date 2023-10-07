#include "NeonVertexArrayObject.h"

namespace Neon
{
	VertexArrayObject::VertexArrayObject()
	{
	}

	VertexArrayObject::~VertexArrayObject()
	{
	}

	void VertexArrayObject::Initialize()
	{
		glGenVertexArrays(1, &id);

		CheckGLError();
	}

	void VertexArrayObject::Terminate()
	{
		glDeleteVertexArrays(1, &id);

		CheckGLError();
	}

	void VertexArrayObject::Bind()
	{
		glBindVertexArray(id);

		CheckGLError();
	}

	void VertexArrayObject::Unbind()
	{
		glBindVertexArray(0);

		CheckGLError();
	}
}
