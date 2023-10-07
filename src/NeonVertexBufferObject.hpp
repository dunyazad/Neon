#include "NeonCommon.h"

namespace Neon
{
	template <class T>
	class VertexBufferObject
	{
	public:
		enum BufferType { VERTEX_BUFFER, NORMAL_BUFFER, INDEX_BUFFER, COLOR_BUFFER, UV_BUFFER };

	public:
		VertexBufferObject(BufferType bufferType)
			: bufferType(bufferType)
		{
		}

		~VertexBufferObject()
		{
		}

		inline unsigned int ID() { return id; }

		void Initialize(unsigned int attributeIndex)
		{
			this->attributeIndex = attributeIndex;

			glGenBuffers(1, &id);

			CheckGLError();
		}

		void Terminate()
		{
			glDeleteBuffers(1, &id);

			CheckGLError();
		}

		void Bind()
		{
			if (bufferType == INDEX_BUFFER)
			{
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id);
			}
			else
			{
				glBindBuffer(GL_ARRAY_BUFFER, id);
			}

			if (dirty)
			{
				Upload();

				dirty = false;
			}
			CheckGLError();
		}

		void Unbind()
		{
			if (bufferType == INDEX_BUFFER)
			{
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			}
			else
			{
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}

			CheckGLError();
		}

		inline size_t Size() { return elements.size(); }
		inline void Clear() { elements.clear(); }

		void AddElement(const T& element)
		{
			elements.push_back(element);

			dirty = true;
		}

		const T& GetElement(int index)
		{
			return elements[index];
		}

		const vector<T>& GetElements() const
		{
			return elements;
		}

		bool SetElement(int index, const T& element)
		{
			if (index >= elements.size())
				return false;

			elements[index] = element;

			dirty = true;

			return true;
		}

		void SetElements(const vector<T>& input)
		{
			elements.resize(input.size());
			memcpy(&elements[0], &input[0], sizeof(T) * input.size());
		}

		void Upload()
		{
			if (elements.size() == 0)
				return;

			if (bufferType == VERTEX_BUFFER)
			{
				glBufferData(GL_ARRAY_BUFFER, sizeof(T) * elements.size(), &elements[0], GL_STATIC_DRAW);
				CheckGLError();

				glVertexAttribPointer(attributeIndex, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
				CheckGLError();

				glEnableVertexAttribArray(attributeIndex);
				CheckGLError();
			}
			else if (bufferType == INDEX_BUFFER)
			{
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(T) * elements.size(), &elements[0], GL_STATIC_DRAW);

				CheckGLError();
			}
			if (bufferType == COLOR_BUFFER)
			{
				glBufferData(GL_ARRAY_BUFFER, sizeof(T) * elements.size(), &elements[0], GL_STATIC_DRAW);
				CheckGLError();

				glVertexAttribPointer(attributeIndex, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
				CheckGLError();

				glEnableVertexAttribArray(attributeIndex);
				CheckGLError();
			}
			else if (bufferType == UV_BUFFER)
			{
				glBufferData(GL_ARRAY_BUFFER, sizeof(T) * elements.size(), &elements[0], GL_STATIC_DRAW);
				CheckGLError();

				glVertexAttribPointer(attributeIndex, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
				CheckGLError();

				glEnableVertexAttribArray(attributeIndex);
				CheckGLError();
			}
			else if (bufferType == NORMAL_BUFFER)
			{
				glBufferData(GL_ARRAY_BUFFER, sizeof(T) * elements.size(), &elements[0], GL_STATIC_DRAW);
				CheckGLError();

				glVertexAttribPointer(attributeIndex, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
				CheckGLError();

				glEnableVertexAttribArray(attributeIndex);
				CheckGLError();
			}
		}

	private:
		bool dirty = true;

		BufferType bufferType;
		unsigned int id;
		unsigned int attributeIndex;
		vector<T> elements;
	};
}