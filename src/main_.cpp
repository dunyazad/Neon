#include <iostream>

#include <Neon/Neon.h>

#include <Neon/CUDA/CUDATest.h>

#include "MiniMath.h"
using namespace MiniMath;

Neon::Scene* g_scene = nullptr;

#pragma region HEM
//namespace HEM
//{
//	typedef unsigned __int64 id_t;
//	const id_t invalid_id = 18446744073709551615;
//
//	struct Vertex
//	{
//		V3 position;
//		id_t half_edge_id = invalid_id;
//	};
//
//	struct HalfEdge
//	{
//		id_t vertex_id = invalid_id;
//		id_t face_id = invalid_id;
//		id_t opposite_half_edge_id = invalid_id;
//		id_t next_half_edge_id = invalid_id;
//	};
//
//	struct Face
//	{
//		id_t half_edge_id = invalid_id;
//	};
//
//	struct KDTreeNode
//	{
//		id_t vertex_id = invalid_id;
//		id_t left_node = invalid_id;
//		id_t right_node = invalid_id;
//	};
//
//	struct Mesh
//	{
//		Vertex* vertices = nullptr;
//		id_t number_of_vertices = invalid_id;
//
//		HalfEdge* halfedges = nullptr;
//		id_t number_of_halfedges = invalid_id;
//
//		Face* faces = nullptr;
//		id_t number_of_faces = invalid_id;
//
//		KDTreeNode* kdtree_nodes = nullptr;
//		id_t number_of_kdtree_nodes = invalid_id;
//
//		id_t root_node = invalid_id;
//	};
//
//	KDTreeNode* GetKDTreeNode(Mesh* mesh, id_t id)
//	{
//		if (id >= mesh->number_of_kdtree_nodes)
//			return nullptr;
//		else
//			return &mesh->kdtree_nodes[id];
//	}
//
//	Vertex* GetVertex(Mesh* mesh, id_t id)
//	{
//		if (id >= mesh->number_of_vertices)
//			return nullptr;
//		else
//			return &mesh->vertices[id];
//	}
//
//	Vertex* GetVertex(Mesh* mesh, const V3& position)
//	{
//		// Todo
//
//		if (invalid_id == mesh->root_node)
//			return nullptr;
//
//		int D = 0;
//		id_t current_node_id = mesh->root_node;
//		id_t last_node_id = current_node_id;
//		float dist = FLT_MAX;
//		while (invalid_id != current_node_id)
//		{
//			auto currentNode = GetKDTreeNode(mesh, current_node_id);
//			auto vertex = GetVertex(mesh, currentNode->vertex_id);
//			//auto current_dist = distance(position, vertex->position);
//
//		}
//
//		return nullptr;
//	}
//
//	HalfEdge* GetHalfEdge(Mesh* mesh, id_t id)
//	{
//		if (id >= mesh->number_of_halfedges)
//			return nullptr;
//		else
//			return &mesh->halfedges[id];
//	}
//
//	Face* GetFace(Mesh* mesh, id_t id)
//	{
//		if (id >= mesh->number_of_faces)
//			return nullptr;
//		else
//			return &mesh->faces[id];
//	}
//
//	id_t AddVertex(const V3& position)
//	{
//		// Todo
//		return invalid_id;
//	}
//}
#pragma endregion

#pragma region Triangulate
//void GetSupraTriangle(const std::vector<glm::vec2>& points, glm::vec2& p0, glm::vec2& p1, glm::vec2& p2)
//{
//	float x = 0.0f;
//	float y = 0.0f;
//	float X = 0.0f;
//	float Y = 0.0f;
//
//	for (auto& p : points)
//	{
//		x = min(x, p.x);
//		X = max(X, p.x);
//		y = min(y, p.y);
//		Y = max(Y, p.y);
//	}
//
//	float cx = (x + X) * 0.5f;
//	float cy = (y + Y) * 0.5f;
//
//	float sx = (x - cx) * 3 + x;
//	float sy = (y - cy) * 3 + y;
//	float sX = (X - cx) * 3 + X;
//	float sY = (Y - cy) * 3 + Y;
//
//	p0 = glm::vec2(sx, sy);
//	p1 = glm::vec2(sX, sy);
//	p2 = glm::vec2(cx, sY);
//}
//
//bool IsPointInTriangle(const glm::vec2& point, const glm::vec2& A, const glm::vec2& B, const glm::vec2& C) {
//	// Compute barycentric coordinates
//	glm::vec2 v0 = B - A;
//	glm::vec2 v1 = C - A;
//	glm::vec2 v2 = point - A;
//
//	float dot00 = glm::dot(v0, v0);
//	float dot01 = glm::dot(v0, v1);
//	float dot02 = glm::dot(v0, v2);
//	float dot11 = glm::dot(v1, v1);
//	float dot12 = glm::dot(v1, v2);
//
//	// Compute barycentric coordinates
//	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
//	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
//	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
//
//	// Check if point is inside the triangle
//	return (u >= 0) && (v >= 0) && (u + v <= 1);
//}
//
//typedef size_t id_t;
//struct Edge
//{
//	id_t i0;
//	id_t i1;
//};
//
//struct Triangle
//{
//	bool valid;
//	id_t i0;
//	id_t i1;
//	id_t i2;
//	glm::vec2 circum_center;
//	float circum_radius;
//};
//
//bool TriangleHasEdge(Triangle* t, Edge* e)
//{
//	return ((e->i0 == t->i0) || (e->i0 == t->i1) || (e->i0 == t->i2)) &&
//		((e->i1 == t->i0) || (e->i1 == t->i1) || (e->i1 == t->i2));
//}
//
//vector<glm::ivec3> Triangulate(const vector<glm::vec3>& pts)
//{
//	vector<glm::vec2> points;
//
//	{ // Copy positions to points and get supra-triangle from pts
//		float x = 0.0f;
//		float y = 0.0f;
//		float X = 0.0f;
//		float Y = 0.0f;
//
//		for (auto& p : pts)
//		{
//			x = min(x, p.x);
//			X = max(X, p.x);
//			y = min(y, p.y);
//			Y = max(Y, p.y);
//
//			points.push_back(glm::vec2(p.x, p.y));
//		}
//
//		float cx = (x + X) * 0.5f;
//		float cy = (y + Y) * 0.5f;
//
//		float sx = (x - cx) * 3 + x;
//		float sy = (y - cy) * 3 + y;
//		float sX = (X - cx) * 3 + X;
//		float sY = (Y - cy) * 3 + Y;
//
//		points.push_back(glm::vec2(sx, sy));
//		points.push_back(glm::vec2(sX, sy));
//		points.push_back(glm::vec2(cx, sY));
//	}
//
//	// points 2d kdtree 생성
//
//	vector<Triangle> triangles;
//
//	// Supra-triangle 생성 후 triangles에 추가
//	// triangles에 추가할때는 circumcenter 와 radius 찾기
//
//
//	triangles.push_back({ true, points.size() - 3, points.size() - 2, points.size() - 1 });
//
//	for (size_t pi = 0; pi < points.size() - 3; pi++)
//	{
//		for (size_t ti = triangles.size() - 1; ti >= 0; ti--)
//		{
//			auto& t = triangles[ti];
//
//		}
//	}
//
//	vector<glm::ivec3> result;
//
//	return result;
//}
#pragma endregion

int main()
{
	Neon::Application app(1280, 1024);
	Neon::URL::ChangeDirectory("..");

	//app.GetWindow()->UseVSync(false);

	int index = 0;

	app.OnInitialize([&]() {
		auto t = Neon::Time("Initialize");

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		//glEnable(GL_CULL_FACE);
		//glCullFace(GL_BACK);
		//glFrontFace(GL_CCW);

		glPointSize(10.0f);

		auto scene = app.CreateScene("Scene/Main");
		g_scene = scene;
		{
			auto toggler = scene->CreateEntity("Scene/Toggler");
			toggler->AddKeyEventHandler([scene](const Neon::KeyEvent& event) {
				if ((GLFW_KEY_0 <= event.key && GLFW_KEY_9 >= event.key) && GLFW_RELEASE == event.action) {
					auto debugEntities = scene->GetDebugEntities();
					if (event.key - GLFW_KEY_0 < debugEntities.size())
					{
						auto entity = debugEntities[(event.key - GLFW_KEY_0 + 1) % debugEntities.size()];

						auto mesh = entity->GetComponent<Neon::Mesh>(0);
						if (nullptr != mesh)
						{
							mesh->ToggleFillMode();
							cout << "Toggle Fill Mode : " << mesh->GetName() << endl;
						}
					}
				}
				});
		}
		
#pragma region Camera
		{
			auto entity = scene->CreateEntity("Entity/Main Camera");
			auto transform = scene->CreateComponent<Neon::Transform>("Transform/Main Camera");
			entity->AddComponent(transform);

			auto camera = scene->CreateComponent<Neon::Camera>("Camera/Main", 1280.0f, 1024.0f);
			entity->AddComponent(camera);
			camera->distance = 2.0f;
			camera->centerPosition = glm::vec3(0.0f, 0.0f, 0.0f);
			scene->SetMainCamera(camera);

			auto cameraManipulator = scene->CreateComponent<Neon::CameraManipulator>("CameraManipulator/Main", entity, camera);
			entity->AddComponent(cameraManipulator);
		}
#pragma endregion

#pragma region Light
		{
			auto entity = scene->CreateEntity("Entity/Main Light");

			auto light = scene->CreateComponent<Neon::Light>("Light/Main");
			entity->AddComponent(light);

			light->AddUpdateHandler([scene, light](double now, double timeDelta) {
				auto camera = scene->GetMainCamera();
				light->position = camera->position;
				light->direction = glm::normalize(camera->centerPosition - camera->position);
				});

			scene->SetMainLight(light);
		}
#pragma endregion

#pragma region Guide Axes
		{
			auto entity = scene->CreateEntity("Entity/Axes");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/Axes");
			entity->AddComponent(mesh);

			mesh->SetDrawingMode(GL_LINES);
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(10.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 10.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 0.0f));
			mesh->AddVertex(glm::vec3(0.0f, 0.0f, 10.0f));

			mesh->AddColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
			mesh->AddColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
			mesh->AddColor(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));

			mesh->AddIndex(0);
			mesh->AddIndex(1);
			mesh->AddIndex(2);
			mesh->AddIndex(3);
			mesh->AddIndex(4);
			mesh->AddIndex(5);

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Color", Neon::URL::Resource("/shader/color.vs"), Neon::URL::Resource("/shader/color.fs"));
			entity->AddComponent(shader);
		}
#pragma endregion

#pragma region Add Points
		//{
//	vector<glm::vec2> points;
//	for (size_t i = 0; i < 1000; i++)
//	{
//		auto x = Neon::RandomReal<float>(-100, 100);
//		auto y = Neon::RandomReal<float>(-100, 100);
//		points.push_back({ x, y });
//	}

//	for (auto& p : points)
//	{
//		scene->Debug("points")->AddPoint({p.x, p.y, 0.0f});
//	}
//}  
#pragma endregion
		{
			std::vector<glm::vec3> points;
			for (double x = -1.0; x <= 1.0; x += 0.1) {
				for (double y = -1.0; y <= 1.0; y += 0.1) {
					for (double z = -1.0; z <= 1.0; z += 0.1) {
						double value = std::sin(x * y * z); // Replace with your own function or use real point cloud data
						glm::vec3 p = glm::vec3(x, y, z + value);
						points.emplace_back(p);

						scene->Debug("Points")->AddPoint(p, {p.x, p.y, p.z, 1.0f});
					}
				}
			}
		}
		});





	app.OnUpdate([&](double now, double timeDelta) {
		//glPointSize(cosf(now * 0.005f) * 10.0f + 10.0f);

		//auto t = Neon::Time("Update");

		//fbo->Bind();

		//glClearColor(0.9f, 0.7f, 0.5f, 1.0f);
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//fbo->Unbind();

		glClearColor(0.3f, 0.5f, 0.7f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		});






	app.OnTerminate([&]() {
		auto t = Neon::Time("Terminate");
		});

	app.Run();

	return 0;
}
