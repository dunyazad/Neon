#include <iostream>

#include <Neon/Neon.h>

int main()
{
	Neon::Application app(1280, 1024);
	app.SetResourceRoot(filesystem::current_path().string() + "/../res");

	Neon::Image* imageB = nullptr;
	Neon::Image* imageC = nullptr;

	app.OnInitialize([&]() {
		auto t = Neon::Time("Initialize");

		auto scene = app.CreateScene("Scene/Main");

		{
			auto entity = scene->CreateEntity("Entity/Main Camera");
			auto camera = scene->CreateComponent<Neon::Camera>("Camera/Main", 1280.0f, 1024.0f);
			scene->SetMainCamera(camera);
			
			entity->AddComponent(camera);
		}

		{
			auto entity = scene->CreateEntity("Entity/triangleA");

			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/triangleA mesh");
			mesh->AddVertex(-0.125f, 0.0f, 0.0f);
			mesh->AddVertex(0.125f, 0.0f, 0.0f);
			mesh->AddVertex(0.0f, 0.5f, 0.0f);

			mesh->AddIndex(0);
			mesh->AddIndex(1);
			mesh->AddIndex(2);

			entity->AddComponent(mesh);
		
			auto shaderFixedColor = scene->CreateComponent<Neon::Shader>("Shader/fixedColor", (app.GetResourceRoot() + "/shader/fixedColor.vs").c_str(), (app.GetResourceRoot() + "/shader/fixedColor.fs").c_str());
			entity->AddComponent(shaderFixedColor);
		}

		{
			auto entity = scene->CreateEntity("Entity/triangleV");

			auto mesh = scene->CreateComponent<Neon::Mesh>("Mesh/triangleV mesh");
			mesh->AddVertex(-0.125f, 0.0f, 0.0f);
			mesh->AddVertex(0.0f, -0.5f, 0.0f);
			mesh->AddVertex(0.125f, 0.0f, 0.0f);

			mesh->AddIndex(0);
			mesh->AddIndex(1);
			mesh->AddIndex(2);

			entity->AddComponent(mesh);

			auto transform = scene->CreateComponent<Neon::Transform>("Transform/triangleV");
			transform->position = glm::vec3(0, -0.1f, 0.0f);
			transform->AddUpdateCallback([transform](float now, float timeDelta) {
				auto angle = now;
				while (angle > 3141.592f) angle -= 3141.592f;
				transform->position = glm::vec3(0, sinf(angle * 0.0001f), 0.0f);
				});

			entity->AddComponent(transform);

			auto shader = scene->GetComponent<Neon::Shader>("Shader/fixedColor");
			entity->AddComponent(shader);
		}

		//{
		//	frame.AddVertex(-0.5f, -0.5f, 0.0f);
		//	frame.AddVertex(0.5f, -0.5f, 0.0f);
		//	frame.AddVertex(0.5f, 0.5f, 0.0f);
		//	frame.AddVertex(-0.5f, 0.5f, 0.0f);

		//	frame.AddIndex(0);
		//	frame.AddIndex(1);
		//	frame.AddIndex(2);

		//	frame.AddIndex(0);
		//	frame.AddIndex(2);
		//	frame.AddIndex(3);

		//	frame.AddUV(0.0f, 0.0f);
		//	frame.AddUV(1.0f, 0.0f);
		//	frame.AddUV(1.0f, 1.0f);
		//	frame.AddUV(0.0f, 1.0f);

		//	textureFrame = new Neon::Texture("frame", 1920, 1080);
		//	fbo = new Neon::FrameBufferObject("frame", textureFrame);
		//}



		{
			auto entity = scene->CreateEntity("owl");

			auto mesh = scene->CreateComponent<Neon::Mesh>("owl mesh");

			mesh->AddVertex(0.5f, 0.0f, 0.0f);
			mesh->AddVertex(0.75f, 0.0f, 0.0f);
			mesh->AddVertex(0.75f, 0.5f, 0.0f);
			mesh->AddVertex(0.5f, 0.5f, 0.0f);

			mesh->AddIndex(0);
			mesh->AddIndex(1);
			mesh->AddIndex(2);

			mesh->AddIndex(0);
			mesh->AddIndex(2);
			mesh->AddIndex(3);

			mesh->AddUV(0.0f, 0.0f);
			mesh->AddUV(1.0f, 0.0f);
			mesh->AddUV(1.0f, 1.0f);
			mesh->AddUV(0.0f, 1.0f);

			imageB = new Neon::Image("Owl.jpg", app.GetResourceRoot() + "/images/Owl.jpg");
			auto texture = new Neon::Texture("Owl", imageB);

			entity->AddComponent(mesh);
			entity->AddComponent(texture);

			auto shaderTexture = scene->CreateComponent<Neon::Shader>("Shader/texture", (app.GetResourceRoot() + "/shader/texture.vs").c_str(), (app.GetResourceRoot() + "/shader/texture.fs").c_str());
			entity->AddComponent(shaderTexture);
		}

		{
			auto entity = scene->CreateEntity("lion");

			auto mesh = scene->CreateComponent<Neon::Mesh>("lion mesh");

			mesh->AddVertex(-0.5f, 0.0f, 0.0f);
			mesh->AddVertex(-0.25f, 0.0f, 0.0f);
			mesh->AddVertex(-0.25f, 0.5f, 0.0f);
			mesh->AddVertex(-0.5f, 0.5f, 0.0f);

			mesh->AddIndex(0);
			mesh->AddIndex(1);
			mesh->AddIndex(2);

			mesh->AddIndex(0);
			mesh->AddIndex(2);
			mesh->AddIndex(3);

			mesh->AddUV(0.0f, 0.0f);
			mesh->AddUV(1.0f, 0.0f);
			mesh->AddUV(1.0f, 1.0f);
			mesh->AddUV(0.0f, 1.0f);

			imageC = new Neon::Image("Lion", app.GetResourceRoot() + "/images/Lion.png");
			auto texture = new Neon::Texture("Lion", imageC);

			entity->AddComponent(mesh);
			entity->AddComponent(texture);
			entity->AddComponent(scene->GetComponent<Neon::Shader>("Shader/texture"));
		}

		});





	app.OnUpdate([&](float now, float timeDelta) {
		//auto t = Neon::Time("Update");

		//fbo->Bind();

		//glClearColor(0.9f, 0.7f, 0.5f, 1.0f);
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//shaderA->use();
		//triangle.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)triangle.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//shaderB->use();
		//owl.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)owl.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//lion.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)lion.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//fbo->Unbind();

		glClearColor(0.3f, 0.5f, 0.7f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//shaderA->Use();
		//triangle.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)triangle.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//shaderB->Use();
		//owl.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)owl.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//lion.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)lion.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		//frame.Bind();
		//glDrawElements(GL_TRIANGLES, (GLsizei)frame.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);
		});






	app.OnTerminate([&]() {
		auto t = Neon::Time("Terminate");

		SAFE_DELETE(imageB);
		SAFE_DELETE(imageC);

		});

	app.Run();

	return 0;
}
