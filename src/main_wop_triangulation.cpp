#include <iostream>

#include <Neon/Neon.h>

Neon::Scene* g_scene = nullptr;

void GetSupraTriangle(const std::vector<glm::vec3>& points, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2)
{
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float X = 0.0f;
	float Y = 0.0f;
	float Z = 0.0f;

	for (auto& p : points)
	{
		x = min(x, p.x);
		X = max(X, p.x);
		y = min(y, p.y);
		Y = max(Y, p.y);
		z = min(z, p.z);
		Z = max(Z, p.z);
	}

	float cx = (x + X) * 0.5f;
	float cy = (y + Y) * 0.5f;
	float cz = (z + Z) * 0.5f;

	float sx = (x - cx) * 3 + x;
	float sy = (y - cy) * 3 + y;
	float sz = (z - cz) * 3 + z;
	float sX = (X - cx) * 3 + X;
	float sY = (Y - cy) * 3 + Y;
	float sZ = (Z - cz) * 3 + Z;

	p0 = glm::vec3(sx, sy, 0.0f);
	p1 = glm::vec3(sX, sy, 0.0f);
	p2 = glm::vec3(cx, sY, 0.0f);
}

bool IsPointInTriangle(const glm::vec3& point, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C) {
	// Compute barycentric coordinates
	glm::vec3 v0 = B - A;
	glm::vec3 v1 = C - A;
	glm::vec3 v2 = point - A;

	float dot00 = glm::dot(v0, v0);
	float dot01 = glm::dot(v0, v1);
	float dot02 = glm::dot(v0, v2);
	float dot11 = glm::dot(v1, v1);
	float dot12 = glm::dot(v1, v2);

	// Compute barycentric coordinates
	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is inside the triangle
	return (u >= 0) && (v >= 0) && (u + v <= 1);
}

bool SplitTriangles(const vector<glm::vec3>& points, vector<glm::ivec4>& triangles)
{
	bool newTriangle = false;

	vector<glm::ivec4> new_triangles;

	for (size_t i = 0; i < triangles.size(); i++)
	{
		auto& t = triangles[i];
		if (1 == t.x)
		{
			auto v0 = points[t.y];
			auto v1 = points[t.z];
			auto v2 = points[t.w];

			for (size_t j = 0; j < points.size(); j++)
			{
				if (j == t.y || j == t.z || j == t.w)
					continue;

				if (IsPointInTriangle(points[j], v0, v1, v2))
				{
					new_triangles.push_back({ 1, t.y, t.z, j });
					new_triangles.push_back({ 1, t.z, t.w, j });
					new_triangles.push_back({ 1, t.w, t.y, j });

					newTriangle = true;

					t.x = 0;
					break;
				}
			}
		}
	}

	for (auto& t : new_triangles)
	{
		triangles.push_back(t);
	}

	return newTriangle;
}


vector<glm::ivec4> Triangulate(const vector<glm::vec3>& points)
{
	vector<glm::vec3> pts(points);
	glm::vec3 sp0, sp1, sp2;
	GetSupraTriangle(points, sp0, sp1, sp2);

	pts.push_back(sp0);
	pts.push_back(sp1);
	pts.push_back(sp2);

	vector<glm::ivec4> triangles;

	triangles.push_back({ 1, pts.size() - 3, pts.size() - 2, pts.size() - 1 });

	int cnt = 0;
	while (SplitTriangles(pts, triangles))
	{
		if (cnt++ == 20) break;
		printf("[%d] triangles.size() : %d\n", cnt, triangles.size());
	}

	for (auto& t : triangles)
	{
		if (1 == t.x)
		{
			auto v0 = pts[t.y];
			auto v1 = pts[t.z];
			auto v2 = pts[t.w];
			g_scene->Debug("Edges")->AddLine(v0, v1, glm::red, glm::red);
			g_scene->Debug("Edges")->AddLine(v1, v2, glm::red, glm::red);
			g_scene->Debug("Edges")->AddLine(v2, v0, glm::red, glm::red);
		}
	}

	return triangles;
}

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

		{
			vector<glm::vec3> points;
			for (size_t i = 0; i < 1000; i++)
			{
				auto x = Neon::RandomReal<float>(-100, 100);
				auto y = Neon::RandomReal<float>(-100, 100);
				points.push_back({ x, y, 0.0f });
			}

			for (auto& p : points)
			{
				scene->Debug("points")->AddPoint(p);
			}

			auto triangles = Triangulate(points);
			for (auto& t : triangles)
			{
				//auto p0 = pts[t.x];
				//auto p1 = pts[t.y];
				//auto p2 = pts[t.z];
				//scene->Debug("Edges")->AddLine(p0, p1, glm::red, glm::red);
				//scene->Debug("Edges")->AddLine(p1, p2, glm::red, glm::red);
				//scene->Debug("Edges")->AddLine(p2, p0, glm::red, glm::red);
			}

			//vector<glm::vec3> pts(points);

			//{ // Get Supra triangle as a initial triangle
			//	glm::vec3 sp0, sp1, sp2;
			//	NeonCUDA::GetSupraTriangle(points, sp0, sp1, sp2);

			//	pts.push_back(sp0);
			//	pts.push_back(sp1);
			//	pts.push_back(sp2);
			//}

			//NeonCUDA::MemoryPoolTest();

	/*		auto triangles = NeonCUDA::Triangulate(points);
			for (auto& t : triangles)
			{
				if (t.x >= pts.size() || t.y >= pts.size() || t.z >= pts.size())
				{
					continue;
				}

				auto p0 = pts[t.x];
				auto p1 = pts[t.y];
				auto p2 = pts[t.z];
				scene->Debug("Edges")->AddLine(p0, p1, glm::red, glm::red);
				scene->Debug("Edges")->AddLine(p1, p2, glm::red, glm::red);
				scene->Debug("Edges")->AddLine(p2, p0, glm::red, glm::red);
			}*/
		}

		/*{
			auto entity = scene->CreateEntity("Mesh");
			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh");
			entity->AddComponent(mesh);

			mesh->FromPLYFile("C:\\Users\\USER\\Desktop\\2023-11-07-Pre-marginline Detection\\2023-11-07-Pre-marginline Detection-lowerjaw.ply");
			mesh->RecalculateFaceNormal();

			auto shader = scene->CreateComponent<Neon::Shader>("Shader/Color", Neon::URL::Resource("/shader/color.vs"), Neon::URL::Resource("/shader/color.fs"));
			entity->AddComponent(shader);
		}*/


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
