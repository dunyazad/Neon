#include <Neon/NeonVertexArrayObject.h>
#include <Neon/NeonVertexBufferObject.hpp>
#include <Neon/Component/NeonMesh.h>
#include <Neon/Component/NeonTexture.h>

#define TINYPLY_IMPLEMENTATION
#include <tinyply/source/tinyply.h>
#include <tinyply/source/example-utils.hpp>

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

	void Mesh::FromPLYFile(const URL& fileURL, float scaleX, float scaleY, float scaleZ)
	{
#pragma region Comment Out
		//vector<glm::vec3> vcPosition;
		//vector<glm::vec3> vcNormal;
		//vector<glm::vec3> vcDiffuse;
		//vector<float> vcReliability;
		//vector<glm::vec3> vcHSV;
		//vector<float> vcAlpha;
		//vector<GLuint> vcMeshIndices;
		//vector<GLuint> vcPointIndices;

		//struct PLY_Info
		//{
		//	bool bIsAscii = false;

		//	bool bExist_NX = false;
		//	bool bExist_NY = false;
		//	bool bExist_NZ = false;

		//	bool bExist_RED = false;
		//	bool bExist_GREEN = false;
		//	bool bExist_BLUE = false;
		//	bool bExist_RELIABILITY = false;
		//	bool bExist_ALPHA = false;

		//	bool bExist_FACE = false;
		//	bool bExist_VERTEXINDEX = false;
		//};

		//PLY_Info stplyInfo;
		//unsigned long ulVertexCount = 0;
		//unsigned long ulTriangleCount = 0;
		//char cBuffer[4096];

		//FILE* fp;
		//fopen_s(&fp, fileURL.path.c_str(), "rb");

		//if (fp == NULL)
		//{
		//	cout << "File doesn't exist." << endl;
		//	return;
		//}

		////Read header
		//while (1)
		//{
		//	fscanf(fp, "%s", cBuffer);

		//	if (strcmp(cBuffer, "ascii") == 0)
		//		stplyInfo.bIsAscii = true;

		//	if (strcmp(cBuffer, "vertex") == 0)
		//	{
		//		fscanf(fp, "%s", cBuffer);
		//		ulVertexCount = std::stoi(cBuffer);

		//		if (ulVertexCount < 1)
		//		{
		//			fclose(fp);
		//			return;
		//		}
		//	}

		//	if (strcmp(cBuffer, "nx") == 0)
		//		stplyInfo.bExist_NX = true;

		//	if (strcmp(cBuffer, "ny") == 0)
		//		stplyInfo.bExist_NY = true;

		//	if (strcmp(cBuffer, "nz") == 0)
		//		stplyInfo.bExist_NZ = true;

		//	if (strcmp(cBuffer, "red") == 0)
		//		stplyInfo.bExist_RED = true;

		//	if (strcmp(cBuffer, "green") == 0)
		//		stplyInfo.bExist_GREEN = true;

		//	if (strcmp(cBuffer, "blue") == 0)
		//		stplyInfo.bExist_BLUE = true;

		//	if (strcmp(cBuffer, "reliability") == 0)
		//		stplyInfo.bExist_RELIABILITY = true;

		//	if (strcmp(cBuffer, "alpha") == 0)
		//		stplyInfo.bExist_ALPHA = true;

		//	if (strcmp(cBuffer, "face") == 0)
		//	{
		//		fscanf(fp, "%s", cBuffer);
		//		ulTriangleCount = std::atoi(cBuffer);

		//		stplyInfo.bExist_FACE = true;
		//	}

		//	if (strcmp(cBuffer, "vertex_indices") == 0)
		//		stplyInfo.bExist_VERTEXINDEX = true;

		//	if (strcmp(cBuffer, "end_header") == 0)
		//		break;
		//}

		////if (bIsMeshType)
		//{
		//	//Unsupported file
		//	if (!stplyInfo.bExist_FACE || !stplyInfo.bExist_VERTEXINDEX)
		//	{
		//		fclose(fp);
		//		cout << "Unsupported file" << endl;
		//	}
		//}
		////else
		//{
		//	//Unsupported file
		//	if (!stplyInfo.bExist_NX || !stplyInfo.bExist_NY || !stplyInfo.bExist_NZ)
		//	{
		//		fclose(fp);
		//		cout << "Unsupported file" << endl;
		//	}
		//}

		////Read vertex info
		//glm::vec3 qvTempPosition;
		//glm::vec3 qvTempNormal;
		//glm::vec3 qvTempDiffuse;
		//float   tempReliability;
		//float   tempAlpha;
		//if (!stplyInfo.bIsAscii)
		//{
		//	long lfilepos = ftell(fp);
		//	fseek(fp, lfilepos + 1, SEEK_SET);

		//	for (int i = 0; i < ulVertexCount; i++)
		//	{
		//		fread(&qvTempPosition, sizeof(float), 3, fp);

		//		if (stplyInfo.bExist_NX && stplyInfo.bExist_NY && stplyInfo.bExist_NZ)
		//			fread(&qvTempNormal, sizeof(float), 3, fp);

		//		if (stplyInfo.bExist_RED && stplyInfo.bExist_GREEN && stplyInfo.bExist_BLUE)
		//		{
		//			unsigned char ucDiffuse = { 0, };

		//			fread(&ucDiffuse, sizeof(unsigned char), 1, fp);
		//			qvTempDiffuse.x = ((float)ucDiffuse / 255.0);

		//			fread(&ucDiffuse, sizeof(unsigned char), 1, fp);
		//			qvTempDiffuse.y = ((float)ucDiffuse / 255.0);

		//			fread(&ucDiffuse, sizeof(unsigned char), 1, fp);
		//			qvTempDiffuse.z = ((float)ucDiffuse / 255.0);

		//			if (stplyInfo.bExist_RELIABILITY) {
		//				fread(&tempReliability, sizeof(float), 1, fp);
		//				vcReliability.push_back(tempReliability);
		//			}
		//			else
		//			{
		//				tempReliability = 0.85;
		//				vcReliability.push_back(tempReliability);

		//			}

		//			if (stplyInfo.bExist_ALPHA) {
		//				fread(&tempAlpha, sizeof(float), 1, fp);
		//				vcAlpha.push_back(tempAlpha);
		//			}
		//			else
		//			{
		//				tempAlpha = 5.;
		//				vcAlpha.push_back(tempAlpha);

		//			}
		//		}

		//		vcPosition.push_back(qvTempPosition);

		//		if (0.0f != qvTempNormal.length())
		//			vcNormal.push_back(qvTempNormal);
		//		else
		//			vcNormal.push_back(glm::vec3(0, 0, 0));
		//		if (0.0f != qvTempDiffuse.length())
		//			vcDiffuse.push_back(qvTempDiffuse);
		//		else
		//			vcDiffuse.push_back(glm::vec3(0.8, 0.8, 0.8));

		//		vcPointIndices.push_back(i);
		//	}

		//	for (unsigned long i = 0; i < ulTriangleCount; i++)
		//	{
		//		unsigned char tempBuffer;
		//		fread(&tempBuffer, sizeof(unsigned char), 1, fp);

		//		int tempindex, tempindex2, tempindex3;
		//		fread(&tempindex, sizeof(int), 1, fp);
		//		vcMeshIndices.push_back(tempindex);

		//		fread(&tempindex2, sizeof(int), 1, fp);
		//		vcMeshIndices.push_back(tempindex2);

		//		fread(&tempindex3, sizeof(int), 1, fp);
		//		vcMeshIndices.push_back(tempindex3);
		//	}
		//}
		//else
		//{
		//	for (unsigned long i = 0; i < ulVertexCount; i++)
		//	{
		//		fscanf(fp, "%s", cBuffer);
		//		qvTempPosition.x = (float)(std::atof(cBuffer));
		//		fscanf(fp, "%s", cBuffer);
		//		qvTempPosition.y = (float)(std::atof(cBuffer));
		//		fscanf(fp, "%s", cBuffer);
		//		qvTempPosition.z = (float)(std::atof(cBuffer));

		//		if (stplyInfo.bExist_NX && stplyInfo.bExist_NY && stplyInfo.bExist_NZ)
		//		{
		//			fscanf(fp, "%s", cBuffer);
		//			qvTempNormal.x = (float)(std::atof(cBuffer));
		//			fscanf(fp, "%s", cBuffer);
		//			qvTempNormal.y = (float)(std::atof(cBuffer));
		//			fscanf(fp, "%s", cBuffer);
		//			qvTempNormal.z = (float)(std::atof(cBuffer));
		//		}

		//		if (stplyInfo.bExist_RED && stplyInfo.bExist_GREEN && stplyInfo.bExist_BLUE)
		//		{
		//			fscanf(fp, "%s", cBuffer);
		//			qvTempDiffuse.x = (float)(std::atof(cBuffer) / 255.0f);
		//			fscanf(fp, "%s", cBuffer);
		//			qvTempDiffuse.y = (float)(std::atof(cBuffer) / 255.0f);
		//			fscanf(fp, "%s", cBuffer);
		//			qvTempDiffuse.z = (float)(std::atof(cBuffer) / 255.0f);

		//			if (stplyInfo.bExist_RELIABILITY)
		//			{
		//				fscanf(fp, "%s", cBuffer);
		//				tempReliability = (float)std::atof(cBuffer);
		//			}
		//			else
		//			{
		//				tempReliability = 0.85;
		//			}

		//			if (stplyInfo.bExist_ALPHA)
		//			{
		//				fscanf(fp, "%s", cBuffer);
		//				tempAlpha = (float)std::atof(cBuffer);
		//			}
		//			else
		//			{
		//				tempAlpha = 5.;
		//			}
		//		}

		//		vcPosition.push_back(qvTempPosition);
		//		vcNormal.push_back(qvTempNormal);

		//		if (0.0f != qvTempDiffuse.length())
		//			vcDiffuse.push_back(qvTempDiffuse);
		//		else
		//			vcDiffuse.push_back(glm::vec3(0.8, 0.8, 0.8));

		//		vcReliability.push_back(tempReliability);
		//		vcAlpha.push_back(tempAlpha);



		//		vcPointIndices.push_back(i);
		//	}
		//	for (unsigned long i = 0; i < ulTriangleCount; i++)
		//	{
		//		fscanf(fp, "%s", cBuffer);
		//		fscanf(fp, "%s", cBuffer);
		//		vcMeshIndices.push_back((GLuint)std::stoi(cBuffer));
		//		fscanf(fp, "%s", cBuffer);
		//		vcMeshIndices.push_back((GLuint)std::stoi(cBuffer));
		//		fscanf(fp, "%s", cBuffer);
		//		vcMeshIndices.push_back((GLuint)std::stoi(cBuffer));
		//	}
		//}

		//fclose(fp);

		////vector<glm::vec3> vcPosition;
		////vector<glm::vec3> vcNormal;
		////vector<glm::vec3> vcDiffuse;
		////vector<float> vcReliability;
		////vector<glm::vec3> vcHSV;
		////vector<float> vcAlpha;

		//for (size_t i = 0; i < ulVertexCount; i++)
		//{
		//	AddIndex(AddVertex(vcPosition[i]));
		//	AddNormal(vcNormal[i]);
		//	AddColor(glm::vec4(vcDiffuse[i], 1.0f));
		//}  
#pragma endregion

		try
		{
			unique_ptr<std::istream> file_stream;
			vector<uint8_t> byte_buffer;

			byte_buffer = read_file_binary(fileURL.path);
			file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));

			if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + fileURL.path);

			file_stream->seekg(0, std::ios::end);
			const float size_mb = file_stream->tellg() * float(1e-6);
			file_stream->seekg(0, std::ios::beg);

			tinyply::PlyFile file;
			file.parse_header(*file_stream);

			std::cout << "\t[ply_header] Type: " << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
			for (const auto& c : file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
			for (const auto& c : file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

			for (const auto& e : file.get_elements())
			{
				std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
				for (const auto& p : e.properties)
				{
					std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
					if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
					std::cout << std::endl;
				}
			}

			// Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers. 
			// See examples below on how to marry your own application-specific data structures with this one. 
			std::shared_ptr<tinyply::PlyData> vertices, normals, colors, texcoords, faces, reliabilities, vcs, alphas;

			// The header information can be used to programmatically extract properties on elements
			// known to exist in the header prior to reading the data. For brevity of this sample, properties 
			// like vertex position are hard-coded: 
			try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
			catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

			try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
			catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

			try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue" }); }
			catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

			try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", "a" }); }
			catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

			try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
			catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

			try { reliabilities = file.request_properties_from_element("vertex", { "reliability" }); }
			catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

			try { vcs = file.request_properties_from_element("vertex", { "cx", "cy", "cz" }); }
			catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

			try { alphas = file.request_properties_from_element("vertex", { "alpha" }); }
			catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

			try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
			catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

			manual_timer read_timer;

			read_timer.start();
			file.read(*file_stream);
			read_timer.stop();

			const float parsing_time = static_cast<float>(read_timer.get()) / 1000.f;
			std::cout << "\tparsing " << size_mb << "mb in " << parsing_time << " seconds [" << (size_mb / parsing_time) << " MBps]" << std::endl;

			if (vertices)   std::cout << "\tRead " << vertices->count << " total vertices " << std::endl;
			if (normals)    std::cout << "\tRead " << normals->count << " total vertex normals " << std::endl;
			if (colors)     std::cout << "\tRead " << colors->count << " total vertex colors " << std::endl;
			if (texcoords)  std::cout << "\tRead " << texcoords->count << " total vertex texcoords " << std::endl;
			if (faces)      std::cout << "\tRead " << faces->count << " total faces (triangles) " << std::endl;
			if (vcs)      std::cout << "\tRead " << vcs->count << " total vcs " << std::endl;
			if (reliabilities)      std::cout << "\tRead " << reliabilities->count << " total reliabilities " << std::endl;
			if (alphas)      std::cout << "\tRead " << alphas->count << " total alphas " << std::endl;
			

			if (vertices)
			{
				const size_t size = vertices->buffer.size_bytes();
				auto buffer = GetVertexBuffer();
				buffer->Clear();
				buffer->Resize(size / sizeof(float3));
				std::memcpy(buffer->Data(), vertices->buffer.get(), size);
			}

			if (normals)
			{
				const size_t size = normals->buffer.size_bytes();
				auto buffer = GetNormalBuffer();
				buffer->Clear();
				buffer->Resize(size / sizeof(float3));
				std::memcpy(buffer->Data(), normals->buffer.get(), size);
			}

			if (colors && alphas)
			{
				unsigned char* alphaBuffer = new unsigned char[alphas->count];
				std::memcpy(alphaBuffer, alphas->buffer.get(), alphas->count * sizeof(unsigned char));

				unsigned char* rgbBuffer = new unsigned char[colors->count * 3];
				std::memcpy(rgbBuffer, colors->buffer.get(), colors->count * 3 * sizeof(unsigned char));

				for (size_t i = 0; i < colors->count; i++)
				{
					float r = ((float)rgbBuffer[i * 3 + 0]) / 255.0f;
					float g = ((float)rgbBuffer[i * 3 + 1]) / 255.0f;
					float b = ((float)rgbBuffer[i * 3 + 2]) / 255.0f;
					float a = ((float)alphaBuffer[i]) / 255.0f;
					AddColor(glm::vec4(r, g, b, a));
				}

				auto buffer = GetColorBuffer();

				delete[] alphaBuffer;
				delete[] rgbBuffer;
			}

			if (faces)
			{
				const size_t size = faces->buffer.size_bytes();
				auto buffer = GetIndexBuffer();
				buffer->Clear();
				buffer->Resize(size / sizeof(GLuint));
				std::memcpy(buffer->Data(), faces->buffer.get(), size);
			}
			else
			{
				for (size_t i = 0; i < vertices->count; i++)
				{
					GetIndexBuffer()->AddElement(i-1);
				}
			}
		}
		catch (const std::exception& e)
		{
			std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
		}
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

	void Mesh::ForEachTriangle(function<void(size_t, GLuint, GLuint, GLuint, glm::vec3, glm::vec3, glm::vec3)> callback)
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
}
