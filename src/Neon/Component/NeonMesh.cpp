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
		aabb.Expand(v);
		return buffer->AddElement(v);
	}

	const glm::vec3& Mesh::GetVertex(size_t index)
	{
		auto buffer = GetVertexBuffer();
		if (index >= buffer->Size())
			return glm::zero<glm::vec3>();

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

	GLuint Mesh::GetIndex(size_t bufferIndex)
	{
		auto buffer = GetIndexBuffer();
		return buffer->GetElement(bufferIndex);
	}

	tuple<int, int, int> Mesh::AddTriangle(GLuint i0, GLuint i1, GLuint i2)
	{
		auto buffer = GetIndexBuffer();
		auto bi0 = (int)buffer->AddElement(i0);
		auto bi1 = (int)buffer->AddElement(i1);
		auto bi2 = (int)buffer->AddElement(i2);
		return make_tuple(bi0, bi1, bi2);
	}

	void Mesh::GetTriangleVertexIndices(size_t triangleIndex, GLuint& i0, GLuint& i1, GLuint& i2)
	{
		auto buffer = GetIndexBuffer();
		i0 = buffer->GetElement(triangleIndex * 3 + 0);
		i1 = buffer->GetElement(triangleIndex * 3 + 1);
		i2 = buffer->GetElement(triangleIndex * 3 + 2);
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

					auto index = AddVertex(glm::vec3(x * scaleX, y * scaleY, z * scaleZ));
					AddIndex((GLuint)index);
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

	void Mesh::ToSTLFile(const URL& fileURL, float scaleX, float scaleY, float scaleZ)
	{
		FILE* fp = nullptr;
		fopen_s(&fp, fileURL.path.c_str(), "wb");

		if (fp != nullptr)
		{
			RecalculateFaceNormal();

			char header[80];
			memset(header, 0, 80);
			char header_string[] = "MeshIO";
			memcpy(header, header_string, 6);
			fwrite(header, 80, 1, fp);

			auto ib = GetIndexBuffer();
			int nof = (int)ib->Size() / 3;
			fwrite(&nof, 4, 1, fp);

			int buffer_index = 0;
			char* buffer = new char[nof * 50];
			memset(buffer, 0, nof * 50);

			for (int i = 0; i < nof; i++)
			{
				auto vi0 = GetIndex(i * 3);
				auto vi1 = GetIndex(i * 3 + 1);
				auto vi2 = GetIndex(i * 3 + 2);

				auto v0 = GetVertex(vi0);
				auto v1 = GetVertex(vi1);
				auto v2 = GetVertex(vi2);

				auto& fn = GetNormal(vi0);
				short dummy = 0;

				v0.x *= scaleX; v0.y *= scaleY; v0.z *= scaleZ;
				v1.x *= scaleX; v1.y *= scaleY; v1.z *= scaleZ;
				v2.x *= scaleX; v2.y *= scaleY; v2.z *= scaleZ;

				memcpy(buffer + buffer_index, &fn, 12); buffer_index += 12;
				memcpy(buffer + buffer_index, &v0, 12); buffer_index += 12;
				memcpy(buffer + buffer_index, &v1, 12); buffer_index += 12;
				memcpy(buffer + buffer_index, &v2, 12); buffer_index += 12;
				memcpy(buffer + buffer_index, &dummy, 2); buffer_index += 2;
			}

			fwrite(buffer, nof * 50, 1, fp);

			delete[] buffer;

			fclose(fp);
		}
	}

	void Mesh::FromBSON(const URL& fileURL, float scaleX, float scaleY, float scaleZ)
	{
		ifstream file(fileURL.path.c_str(), ios::binary);
		if (file.is_open())
		{
			file.seekg(0, ios::end);
			auto fileSize = file.tellg();
			file.seekg(0, ios::beg);

			vector<uint8_t> fileContents(static_cast<size_t>(fileSize));
			file.read(reinterpret_cast<char*>(fileContents.data()), fileSize);
			file.close();

			json json = json::from_bson(fileContents);

			{
				auto nov = json["number of vertices"].get<size_t>();
				if (0 < nov)
				{
					printf("number of vertices : %lld\n", nov);

					auto vs = json["vertices"].get<vector<float>>();
					for (size_t i = 0; i < nov; i++)
					{
						auto x = vs[i * 3 + 0] * scaleX;
						auto y = vs[i * 3 + 1] * scaleY;
						auto z = vs[i * 3 + 2] * scaleZ;

						AddVertex({ x, y, z });
					}
				}
			}

			{
				auto non = json["number of normals"].get<size_t>();
				if (0 < non)
				{
					auto ns = json["normals"].get<vector<float>>();
					for (size_t i = 0; i < non; i++)
					{
						auto x = ns[i * 3 + 0];
						auto y = ns[i * 3 + 1];
						auto z = ns[i * 3 + 2];

						AddNormal({ x, y, z });
					}
				}
			}

			{
				auto noc = json["number of colors"].get<size_t>();
				if (0 < noc)
				{
					auto cs = json["colors"].get<vector<unsigned char>>();
					for (size_t i = 0; i < noc; i++)
					{
						auto r = cs[i * 3 + 0];
						auto g = cs[i * 3 + 1];
						auto b = cs[i * 3 + 2];

						AddColor({ r, g, b, 1.0f });
					}
				}
			}

			{
				auto not = json["number of triangles"].get<size_t>();
				if (0 < not)
				{
					auto is = json["triangles"].get<vector<tuple<unsigned int, unsigned int, unsigned int>>>();
					for (size_t i = 0; i < not; i++)
					{
						AddIndex(get<0>(is[i]));
						AddIndex(get<1>(is[i]));
						AddIndex(get<2>(is[i]));
					}
				}
			}

			{
				auto nol = json["number of lines"].get<size_t>();
				if (0 < nol)
				{
					auto is = json["lines"].get<vector<tuple<unsigned int, unsigned int>>>();
					for (size_t i = 0; i < nol; i++)
					{
						auto& i0 = get<0>(is[i]);
						auto& i1 = get<1>(is[i]);

						AddIndex(i0);
						AddIndex(i1);

						printf("i0 : %d, i1 : %d\n", i0, i1);
					}
				}
			}

			//{
			//	auto nov = json["number of vertices"].get<size_t>();
			//	AddIndex(nov - 4);
			//	AddIndex(nov - 3);
			//	AddIndex(nov - 3);
			//	AddIndex(nov - 2);
			//	AddIndex(nov - 2);
			//	AddIndex(nov - 1);
			//}
		}
	}

	inline std::vector<uint8_t> read_file_binary(const std::string& pathToFile)
	{
		std::ifstream file(pathToFile, std::ios::binary);
		std::vector<uint8_t> fileBufferBytes;

		if (file.is_open())
		{
			file.seekg(0, std::ios::end);
			size_t sizeBytes = file.tellg();
			file.seekg(0, std::ios::beg);
			fileBufferBytes.resize(sizeBytes);
			if (file.read((char*)fileBufferBytes.data(), sizeBytes)) return fileBufferBytes;
		}
		else throw std::runtime_error("could not open binary ifstream to path " + pathToFile);
		return fileBufferBytes;
	}

	/*
	class PLYDataType
	{
	public:
		enum Type { CHAR, UCHAR, SHORT, USHORT, INT, UINT, FLOAT, DOUBLE, USER_DEFINED, NONE };
		Type type;
		int dataSize = 0;
		PLYDataType(Type type)
		{
			type = type;

		}
		PLYDataType(const string& typeName) {}

	public:
		static Type ToType(const string& typeName)
		{
			if ("char" == typeName) return CHAR;
			else if ("uchar" == typeName) return UCHAR;
			else if ("short" == typeName) return SHORT;
			else if ("ushort" == typeName) return USHORT;
			else if ("int" == typeName) return INT;
			else if ("unsigned int" == typeName) return UINT;
			else if ("float" == typeName) return FLOAT;
			else if ("double" == typeName) return DOUBLE;
			else if ("user_defined" == typeName) return USER_DEFINED;
			else if ("none" == typeName) return NONE;
		}

		static string ToTypeName(Type type)
		{
			if (CHAR == type) return "char";
			else if (UCHAR == type) return "unsigned char";
			else if (SHORT == type) return "short";
			else if (USHORT == type) return "unsigned short";
			else if (INT == type) return "int";
			else if (UINT == type) return "unsigned int";
			else if (FLOAT == type) return "float";
			else if (DOUBLE == type) return "double";
			else if (USER_DEFINED == type) return "user_defined";
			else if (NONE == type) return "none";
		}

		static int DataSize(Type type)
		{
			if (CHAR == type) return sizeof(char);
			else if (UCHAR == type) return sizeof(unsigned char);
			else if (SHORT == type) return sizeof(short);
			else if (USHORT == type) return sizeof(unsigned short);
			else if (INT == type) return sizeof(int);
			else if (UINT == type) return sizeof(unsigned int);
			else if (FLOAT == type) return sizeof(float);
			else if (DOUBLE == type) return sizeof(double);
			else if (USER_DEFINED == type) return 0;
			else if (NONE == type) return 0;
		}

		static int DataSize(const string& typeName)
		{
			return DataSize(ToType(typeName));
		}
	};

	class PLYProperty
	{
	public:
		enum PropertyType {
			CHAR, CHAR_LIST,
			UCHAR, UCHAR_LIST,
			SHORT, SHORT_LIST,
			USHORT, USHORT_LIST,
			INT, INT_LIST,
			UINT, UINT_LIST,
			FLOAT, FLOAT_LIST,
			DOUBLE, DOUBLE_LIST,
		};

	protected:
		string name;
	};

	class PLYElement
	{
	public:
		PLYElement() {}
		virtual ~PLYElement() {}

		void Parse(const string& line)
		{

		}

	protected:
		string name;
		vector<tuple<string, PLYProperty>> properties;
		vector<char> data;
	};

	class PLYElementGroup
	{
	public:
		PLYElementGroup() {}
		virtual ~PLYElementGroup() {}

		void Parse(ifstream& ifs)
		{
			for (size_t i = 0; i < elements.size(); i++)
			{
				string line;
				std::getline(ifs, line);
				elements[i].Parse(line);
			}
		}

	protected:
		vector<tuple<string, PLYElement>> elements;
	};

	class PLYFormat
	{
		vector<PLYElement> elements;
		bool isBinary = false;

	public:
		PLYFormat() {}
		virtual ~PLYFormat() {}

		inline const vector<PLYElement>& GetElements() { return elements; };

		PLYElement* AddElement(const string& name, size_t count)
		{
			for (auto& element : elements)
			{
				if (name == element.GetName())
					return nullptr;
			}

			elements.emplace_back(name, count);
			return &elements.back();
		}

		void AddProperty(PLYElement* element, const string& dataType, const string& name)
		{
			element->AddProperty(dataType, name);
		}

		void AddPropertyList(PLYElement* element, const string& countType, const string& dataType, const string& name)
		{
			element->AddPropertyList(countType, dataType, name);
		}

		void Read(const URL& fileURL)
		{
			ifstream ifs(fileURL.path);
			if (ifs.is_open() == false)
				return;

			string line;
			std::getline(ifs, line);
			if ("ply" != line) return;

			PLYElement* recentElement = nullptr;

			// Read Header
			while (std::getline(ifs, line))
			{
				stringstream ss(line);
				string word;

				ss >> word;

				if ("format" == word)
				{
					ss >> word;
					if ("binary_little_endian" == word || "binary_big_endian" == word)
						isBinary = true;
				}

				if ("comment" == word)
					continue;

				if ("element" == word)
				{
					string name;
					ss >> name;
					size_t count;
					ss >> count;

					recentElement = AddElement(name, count);
				}

				if ("property" == word)
				{
					string dataType;
					ss >> dataType;

					if ("list" != dataType)
					{
						string name;
						ss >> name;

						if (nullptr != recentElement)
						{
							AddProperty(recentElement, dataType, name);
						}
					}
					else
					{
						string countType;
						ss >> countType;
						string dataType;
						ss >> dataType;
						string name;
						ss >> name;

						if (nullptr != recentElement)
						{
							AddPropertyList(recentElement, countType, dataType, name);
						}
					}
				}

				if ("end_header" == word)
					break;
			}

			// Read Data
			for (auto& element : elements)
			{
				element.Parse(ifs);
			}
		}

		void Write(const URL& fileURL)
		{
		}
	};
	*/

	void Mesh::FromPLYFile(const URL& fileURL, float scaleX, float scaleY, float scaleZ)
	{
		/*auto ply = PLYFormat();
		ply.Read(fileURL);

		for (auto& element : ply.GetElements())
		{
			printf("%s : %d\n", element.GetName().c_str(), element.GetCount());
		}*/
	}

	void Mesh::FromXYZWFile(const URL& fileURL, float scaleX, float scaleY, float scaleZ)
	{
		auto _T = Neon::Time("FromXYZWFile()");

		FILE* fp = nullptr;
		auto err = fopen_s(&fp, fileURL.path.c_str(), "rb");
		if (0 != err)
		{
			printf("[Deserialize] File \"%s\" open failed.", fileURL.path.c_str());
		}

		char buffer[1024];
		memset(buffer, 0, 1024);
		auto line = fgets(buffer, 1024, fp);
		if (0 != strcmp(line, "XYZW\n"))
			return;

		line = fgets(buffer, 1024, fp);
		while ('#' == line[0])
		{
			line = fgets(buffer, 1024, fp);
		}

		size_t vertexCount = 0;
		size_t triangleCount = 0;
		sscanf_s(line, "%d %d", &vertexCount, &triangleCount);

		printf("vertexCount : %d, triangleCount : %d\n", vertexCount, triangleCount);

		for (size_t i = 0; i < vertexCount; i++)
		{
			line = fgets(buffer, 1024, fp);
			if (nullptr != line)
			{
				if ('#' == line[0])
				{
					i--;
					continue;
				}
				else
				{
					float x, y, z, w;
					sscanf_s(line, "%f %f %f %f\n", &x, &y, &z, &w);

					AddVertex({ x, y, z });
				}
			}
		}

		for (size_t i = 0; i < triangleCount; i++)
		{
			line = fgets(buffer, 1024, fp);
			if ('#' == line[0])
			{
				i--;
				continue;
			}
			else
			{
				size_t count, i0, i1, i2;
				int r, g, b;
				sscanf_s(line, "%d %d %d %d %d %d %d\n", &count, &i0, &i1, &i2, &r, &g, &b);

				AddIndex(i0);
				AddIndex(i1);
				AddIndex(i2);

				AddColor(glm::vec4((float)(r) / 255.f, (float)(g) / 255.f, (float)(b) / 255.f, 1.0f));
				AddColor(glm::vec4((float)(r) / 255.f, (float)(g) / 255.f, (float)(b) / 255.f, 1.0f));
				AddColor(glm::vec4((float)(r) / 255.f, (float)(g) / 255.f, (float)(b) / 255.f, 1.0f));
			}
		}

		//return false;

		//while (nullptr != line)
		//{
		//	printf("%s", line);

		//	line = fgets(buffer, 1024, fp);
		//}

		fclose(fp);
	}

	void Mesh::RecalculateFaceNormal()
	{
		auto nov = GetVertexBuffer()->Size();
		GetNormalBuffer()->Resize(nov);

		auto noi = GetIndexBuffer()->Size();
		for (int i = 0; i < noi / 3; i++)
		{
			auto i0 = GetIndex(i * 3 + 0);
			auto i1 = GetIndex(i * 3 + 1);
			auto i2 = GetIndex(i * 3 + 2);

			auto v0 = GetVertex(i0);
			auto v1 = GetVertex(i1);
			auto v2 = GetVertex(i2);

			auto normal = glm::normalize(glm::cross(glm::normalize(v1 - v0), glm::normalize(v2 - v0)));
			SetNormal(i0, normal);
			SetNormal(i1, normal);
			SetNormal(i2, normal);
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
			auto i0 = ib->GetElement(i * 3 + 0);
			auto i1 = ib->GetElement(i * 3 + 1);
			auto i2 = ib->GetElement(i * 3 + 2);

			auto v0 = GetVertex(i0);
			auto v1 = GetVertex(i1);
			auto v2 = GetVertex(i2);

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

	void Mesh::ForEachTriangle(function<void(size_t, GLuint, GLuint, GLuint, glm::vec3&, glm::vec3&, glm::vec3&)> callback)
	{
		auto ib = GetIndexBuffer();
		auto noi = ib->Size();
		for (size_t i = 0; i < noi / 3; i++)
		{
			auto i0 = ib->GetElement(i * 3 + 0);
			auto i1 = ib->GetElement(i * 3 + 1);
			auto i2 = ib->GetElement(i * 3 + 2);

			auto v0 = GetVertex(i0);
			auto v1 = GetVertex(i1);
			auto v2 = GetVertex(i2);

			callback(i, i0, i1, i2, v0, v1, v2);
		}
	}

	void Mesh::Bake(const glm::mat4& transformMatrix)
	{
		aabb.Clear();

		auto vertexBuffer = GetVertexBuffer();
		if (nullptr != vertexBuffer)
		{
			for (auto& v : vertexBuffer->GetElements())
			{
				if (FLT_VALID(v.x) && FLT_VALID(v.y) && FLT_VALID(v.z))
				{
					v = glm::vec3(transformMatrix * glm::vec4(v.x, v.y, v.z, 1.0f));

					aabb.Expand(v);
				}
				//else
				//{
				//	v = glm::zero<glm::vec3>();
				//}
			}
			vertexBuffer->SetDirty();
		}

		auto normalBuffer = GetNormalBuffer();
		if (nullptr != normalBuffer)
		{
			for (auto& n : normalBuffer->GetElements())
			{
				if (FLT_VALID(n.x) && FLT_VALID(n.y) && FLT_VALID(n.z))
				{
					n = glm::vec3(transformMatrix * glm::vec4(n.x, n.y, n.z, 1.0f));
				}
			}
			normalBuffer->SetDirty();
		}
	}
}
