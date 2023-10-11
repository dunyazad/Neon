#include <Neon/NeonVertexArrayObject.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonTexture.h>

namespace Neon
{
	Mesh::Mesh(const string& name)
		: ComponentBase(name)
	{
		vao = new VertexArrayObject();
	} 

	Mesh::~Mesh()
	{
		Unbind();

		for (auto& kvp : bufferObjects)
		{
			SAFE_DELETE(kvp.second);
		}

		SAFE_DELETE(vao);
	}

	VertexBufferObject<float>* Mesh::GetVertexBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::VERTEX_BUFFER))
		{
			auto buffer = new VertexBufferObject<float>(VertexBufferObjectBase::BufferType::VERTEX_BUFFER, VertexBufferObjectBase::BufferType::VERTEX_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::VERTEX_BUFFER] = buffer;
		}
		return (VertexBufferObject<float>*)bufferObjects[VertexBufferObjectBase::BufferType::VERTEX_BUFFER];
	}

	VertexBufferObject<float>* Mesh::GetNormalBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::NORMAL_BUFFER))
		{
			auto buffer = new VertexBufferObject<float>(VertexBufferObjectBase::BufferType::NORMAL_BUFFER, VertexBufferObjectBase::BufferType::NORMAL_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::NORMAL_BUFFER] = buffer;
		}

		return (VertexBufferObject<float>*)bufferObjects[VertexBufferObjectBase::BufferType::NORMAL_BUFFER];
	}

	VertexBufferObject<GLuint>* Mesh::GetIndexBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::INDEX_BUFFER))
		{
			auto buffer = new VertexBufferObject<GLuint>(VertexBufferObjectBase::BufferType::INDEX_BUFFER, VertexBufferObjectBase::BufferType::INDEX_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::INDEX_BUFFER] = buffer;
		}

		return (VertexBufferObject<GLuint>*)bufferObjects[VertexBufferObjectBase::BufferType::INDEX_BUFFER];
	}

	VertexBufferObject<float>* Mesh::GetColorBuffer() 
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::COLOR_BUFFER))
		{
			auto buffer = new VertexBufferObject<float>(VertexBufferObjectBase::BufferType::COLOR_BUFFER, VertexBufferObjectBase::BufferType::COLOR_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::COLOR_BUFFER] = buffer;
		}

		return (VertexBufferObject<float>*)bufferObjects[VertexBufferObjectBase::BufferType::COLOR_BUFFER];
	}

	VertexBufferObject<float>* Mesh::GetUVBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::UV_BUFFER))
		{
			auto buffer = new VertexBufferObject<float>(VertexBufferObjectBase::BufferType::UV_BUFFER, VertexBufferObjectBase::BufferType::UV_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::UV_BUFFER] = buffer;
		}

		return (VertexBufferObject<float>*)bufferObjects[VertexBufferObjectBase::BufferType::UV_BUFFER];
	}

	void Mesh::AddVertex(float x, float y, float z)
	{
		auto buffer = GetVertexBuffer();
		buffer->AddElement(x);
		buffer->AddElement(y);
		buffer->AddElement(z);
	}

	void Mesh::GetVertex(int index, float& x, float& y, float& z)
	{
		auto buffer = GetVertexBuffer();
		x = buffer->GetElement(index * 3 + 0);
		y = buffer->GetElement(index * 3 + 1);
		z = buffer->GetElement(index * 3 + 2);
	}

	void Mesh::SetVertex(int index, float x, float y, float z)
	{
		auto buffer = GetVertexBuffer();
		buffer->SetElement(index * 3 + 0, x);
		buffer->SetElement(index * 3 + 1, y);
		buffer->SetElement(index * 3 + 2, z);
	}

	void Mesh::AddNormal(float x, float y, float z)
	{
		auto buffer = GetNormalBuffer();
		buffer->AddElement(x);
		buffer->AddElement(y);
		buffer->AddElement(z);
	}

	void Mesh::GetNormal(int index, float& x, float& y, float& z)
	{
		auto buffer = GetNormalBuffer();
		x = buffer->GetElement(index * 3 + 0);
		y = buffer->GetElement(index * 3 + 1);
		z = buffer->GetElement(index * 3 + 2);
	}
	
	void Mesh::SetNormal(int index, float x, float y, float z)
	{
		auto buffer = GetNormalBuffer();
		buffer->SetElement(index * 3 + 0, x);
		buffer->SetElement(index * 3 + 1, y);
		buffer->SetElement(index * 3 + 2, z);
	}

	void Mesh::AddIndex(GLuint index)
	{
		auto buffer = GetIndexBuffer();
		buffer->AddElement(index);
	}

	void Mesh::AddColor(float r, float g, float b, float a)
	{
		auto buffer = GetColorBuffer();
		buffer->AddElement(r);
		buffer->AddElement(g);
		buffer->AddElement(b);
		buffer->AddElement(a);
	}
	
	void Mesh::AddUV(float u, float v)
	{
		auto buffer = GetUVBuffer();
		buffer->AddElement(u);
		buffer->AddElement(v);
	}

	void Mesh::Bind()
	{
		vao->Bind();

		for (auto& kvp : bufferObjects)
		{
			kvp.second->Bind();
			kvp.second->Upload();
		}
	}

	void Mesh::Unbind()
	{
		vao->Unbind();

		for (auto& kvp : bufferObjects)
		{
			kvp.second->Unbind();
		}
	}

	void Mesh::FromSTLFile(const string& filePath, float scaleX, float scaleY, float scaleZ)
	{
		ifstream ifs(filePath);
		if (ifs.is_open() == false)
			return;

		string solid = "      ";
		ifs.read(&solid[0], 6);

		if (solid == "solid ")
		{
			ifs.close();
			
			ifstream ifs(filePath);
			stringstream buffer;
			buffer << ifs.rdbuf();

			glm::vec3 fn;
			int vertex_index = 0;

			string line;
			while (buffer.good())
			{
				getline(buffer, line);
				if (line.empty())
					continue;

				auto words = split(line, " \t");
				if (words[0] == "facet")
				{
					if (words[1] == "normal")
					{
						fn.x = safe_stof(words[2]);
						fn.y = safe_stof(words[3]);
						fn.z = safe_stof(words[4]);

						AddNormal(fn.x, fn.y, fn.z);
					}
				}
				else if (words[0] == "vertex")
				{
					float x = safe_stof(words[1]);
					float y = safe_stof(words[2]);
					float z = safe_stof(words[3]);

					AddVertex(x * scaleX, y * scaleY, z * scaleZ);
					AddIndex(vertex_index++);
				}
				else if (words[0] == "endfacet")
				{
					vertex_index = 0;
				}
			}
		}
		else
		{
			ifs.close();
			
			FILE* fp = nullptr;
			fopen_s(&fp, filePath.c_str(), "rb");
			if (fp != nullptr)
			{
				char header[80];
				memset(header, 0, 80);
				fread_s(header, 80, 80, 1, fp);

				int nof = 0;
				fread_s(&nof, 4, 4, 1, fp);

				int vertex_index = 0;
				for (size_t i = 0; i < nof; i++)
				{
					glm::vec3 fn, v0, v1, v2;
					short dummy;

					fread_s(&fn, 12, 12, 1, fp);
					fread_s(&v0, 12, 12, 1, fp);
					fread_s(&v1, 12, 12, 1, fp);
					fread_s(&v2, 12, 12, 1, fp);
					fread_s(&dummy, 2, 2, 1, fp);

					AddVertex(v0.x * scaleX, v0.y * scaleY, v0.z * scaleZ);
					AddNormal(fn.x, fn.y, fn.z);
					AddIndex(vertex_index++);

					AddVertex(v1.x * scaleX, v1.y * scaleY, v1.z * scaleZ);
					AddNormal(fn.x, fn.y, fn.z);
					AddIndex(vertex_index++);

					AddVertex(v2.x * scaleX, v2.y * scaleY, v2.z * scaleZ);
					AddNormal(fn.x, fn.y, fn.z);
					AddIndex(vertex_index++);
				}
			}
		}
	}

	void Mesh::RecalculateFaceNormal()
	{
		auto noi = GetIndexBuffer()->Size();
		for (size_t i = 0; i < noi / 3; i++)
		{
			glm::vec3 v0, v1, v2;
			GetVertex(i * 3 + 0, v0.x, v0.y, v0.z);
			GetVertex(i * 3 + 1, v1.x, v1.y, v1.z);
			GetVertex(i * 3 + 2, v2.x, v2.y, v2.z);

			auto normal = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));
			SetNormal(i * 3 + 0, normal.x, normal.y, normal.z);
			SetNormal(i * 3 + 1, normal.x, normal.y, normal.z);
			SetNormal(i * 3 + 2, normal.x, normal.y, normal.z);
		}
	}
}
