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

	void Mesh::Clear()
	{
		for (auto& kvp : bufferObjects)
		{
			kvp.second->Clear();
		}
	}

	VertexBufferObject<glm::vec3>* Mesh::GetVertexBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::VERTEX_BUFFER))
		{
			auto buffer = new VertexBufferObject<glm::vec3>(VertexBufferObjectBase::BufferType::VERTEX_BUFFER, VertexBufferObjectBase::BufferType::VERTEX_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::VERTEX_BUFFER] = buffer;
		}
		return (VertexBufferObject<glm::vec3>*)bufferObjects[VertexBufferObjectBase::BufferType::VERTEX_BUFFER];
	}

	VertexBufferObject<glm::vec3>* Mesh::GetNormalBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::NORMAL_BUFFER))
		{
			auto buffer = new VertexBufferObject<glm::vec3>(VertexBufferObjectBase::BufferType::NORMAL_BUFFER, VertexBufferObjectBase::BufferType::NORMAL_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::NORMAL_BUFFER] = buffer;
		}

		return (VertexBufferObject<glm::vec3>*)bufferObjects[VertexBufferObjectBase::BufferType::NORMAL_BUFFER];
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

	VertexBufferObject<glm::vec4>* Mesh::GetColorBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::COLOR_BUFFER))
		{
			auto buffer = new VertexBufferObject<glm::vec4>(VertexBufferObjectBase::BufferType::COLOR_BUFFER, VertexBufferObjectBase::BufferType::COLOR_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::COLOR_BUFFER] = buffer;
		}

		return (VertexBufferObject<glm::vec4>*)bufferObjects[VertexBufferObjectBase::BufferType::COLOR_BUFFER];
	}

	VertexBufferObject<glm::vec2>* Mesh::GetUVBuffer()
	{
		if (0 == bufferObjects.count(VertexBufferObjectBase::BufferType::UV_BUFFER))
		{
			auto buffer = new VertexBufferObject<glm::vec2>(VertexBufferObjectBase::BufferType::UV_BUFFER, VertexBufferObjectBase::BufferType::UV_BUFFER);
			bufferObjects[VertexBufferObjectBase::BufferType::UV_BUFFER] = buffer;
		}

		return (VertexBufferObject<glm::vec2>*)bufferObjects[VertexBufferObjectBase::BufferType::UV_BUFFER];
	}

	size_t Mesh::AddVertex(const glm::vec3& v)
	{
		auto buffer = GetVertexBuffer();
		return buffer->AddElement(v);
	}

	const glm::vec3& Mesh::GetVertex(size_t index)
	{
		auto buffer = GetVertexBuffer();
		return buffer->GetElement(index);
	}

	void Mesh::SetVertex(size_t index, const glm::vec3& v)
	{
		auto buffer = GetVertexBuffer();
		buffer->SetElement(index, v);
	}

	size_t Mesh::AddNormal(const glm::vec3& n)
	{
		auto buffer = GetNormalBuffer();
		return buffer->AddElement(n);
	}

	const glm::vec3& Mesh::GetNormal(size_t index)
	{
		auto buffer = GetNormalBuffer();
		return buffer->GetElement(index);
	}

	void Mesh::SetNormal(size_t index, const glm::vec3& n)
	{
		auto buffer = GetNormalBuffer();
		buffer->SetElement(index, n);
	}

	size_t Mesh::AddIndex(GLuint index)
	{
		auto buffer = GetIndexBuffer();
		return buffer->AddElement(index);
	}

	void Mesh::GetIndex(size_t bufferIndex, GLuint& index)
	{
		auto buffer = GetIndexBuffer();
		index = buffer->GetElement(bufferIndex);
	}

	size_t Mesh::AddColor(const glm::vec4& c)
	{
		auto buffer = GetColorBuffer();
		return buffer->AddElement(c);
	}

	const glm::vec4& Mesh::GetColor(size_t index)
	{
		auto buffer = GetColorBuffer();
		return buffer->GetElement(index);
	}

	void Mesh::SetColor(size_t index, const glm::vec4& c)
	{
		auto buffer = GetColorBuffer();
		buffer->SetElement(index, c);
	}

	size_t Mesh::AddUV(const glm::vec2& uv)
	{
		auto buffer = GetUVBuffer();
		return buffer->AddElement(uv);
	}

	const glm::vec2& Mesh::GetUV(size_t index)
	{
		auto buffer = GetUVBuffer();
		return buffer->GetElement(index);
	}

	void Mesh::SetUV(size_t index, const glm::vec2& uv)
	{
		auto buffer = GetUVBuffer();
		buffer->SetElement(index, uv);
	}

	void Mesh::Bind()
	{
		vao->Bind();

		for (auto& kvp : bufferObjects)
		{
			kvp.second->Bind();
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

	void Mesh::FromSTLFile(const URL& fileURL, float scaleX, float scaleY, float scaleZ)
	{
		ifstream ifs(fileURL.path);
		if (ifs.is_open() == false)
			return;

		string solid = "      ";
		ifs.read(&solid[0], 6);

		if (solid == "solid ")
		{
			ifs.close();
			
			ifstream ifs(fileURL.path);
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

						AddNormal(fn);
					}
				}
				else if (words[0] == "vertex")
				{
					float x = safe_stof(words[1]);
					float y = safe_stof(words[2]);
					float z = safe_stof(words[3]);

					AddVertex(glm::vec3(x * scaleX, y * scaleY, z * scaleZ));
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
			fopen_s(&fp, fileURL.path.c_str(), "rb");
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

					AddVertex(glm::vec3(v0.x * scaleX, v0.y * scaleY, v0.z * scaleZ));
					AddNormal(fn);
					AddIndex(vertex_index++);

					AddVertex(glm::vec3(v1.x * scaleX, v1.y * scaleY, v1.z * scaleZ));
					AddNormal(fn);
					AddIndex(vertex_index++);

					AddVertex(glm::vec3(v2.x * scaleX, v2.y * scaleY, v2.z * scaleZ));
					AddNormal(fn);
					AddIndex(vertex_index++);
				}
			}
		}
	}

	void Mesh::RecalculateFaceNormal()
	{
		auto noi = GetIndexBuffer()->Size();
		for (int i = 0; i < noi / 3; i++)
		{
			auto v0 = GetVertex(i * 3 + 0);
			auto v1 = GetVertex(i * 3 + 1);
			auto v2 = GetVertex(i * 3 + 2);

			auto normal = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));
			SetNormal(i * 3 + 0, normal);
			SetNormal(i * 3 + 1, normal);
			SetNormal(i * 3 + 2, normal);
		}
	}

	void Mesh::FillColor(const glm::vec4& color)
	{
		auto nov = GetVertexBuffer()->Size();
		GetColorBuffer()->Clear();
		for (size_t i = 0; i < nov; i++)
		{
			AddColor(color);
		}
	}

	bool Mesh::Pick(const Ray& ray, glm::vec3& intersection, size_t& faceIndex)
	{
		vector<pair<float, int>> pickedFaceIndices;
		auto ib = GetIndexBuffer();
		auto noi = ib->Size();
		for (size_t i = 0; i < noi / 3; i++)
		{
			auto v0 = GetVertex(i * 3 + 0);
			auto v1 = GetVertex(i * 3 + 1);
			auto v2 = GetVertex(i * 3 + 2);

			glm::vec2 baricenter;
			float distance = 0.0f;
			if (glm::intersectRayTriangle(ray.origin, ray.direction, v0, v1, v2, baricenter, distance))
			{
				if (distance > 0) {
					pickedFaceIndices.push_back(make_pair(distance, (int)i));
				}
			}
		}

		if (0 < pickedFaceIndices.size())
		{
			struct PickedFacesLess {
				inline bool operator() (const tuple<float, int>& a, const tuple<float, int>& b) {
					return get<0>(a) < get<0>(b);
				}
			};

			sort(pickedFaceIndices.begin(), pickedFaceIndices.end(), PickedFacesLess());

			intersection = ray.origin + ray.direction * pickedFaceIndices.front().first;
			faceIndex = pickedFaceIndices.front().second;
			return true;
		}
		else
		{
			return false;
		}
	}
}
