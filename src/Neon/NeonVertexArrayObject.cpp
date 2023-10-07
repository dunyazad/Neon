#include <Neon/NeonVertexArrayObject.h>

namespace Neon
{
	VertexArrayObject::VertexArrayObject()
	{
		glGenVertexArrays(1, &id);

		CheckGLError();
	}

	VertexArrayObject::~VertexArrayObject()
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
