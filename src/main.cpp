#include <iostream>

#include <Neon/Neon.h>

int main()
{	
	Neon::Application app(1280, 1024);
	app.SetResourceRoot(filesystem::current_path().string() + "/../res");

	Neon::Shader* shaderA = nullptr;
	Neon::Shader* shaderB = nullptr;

	Neon::Image* imageB = nullptr;
	Neon::Image* imageC = nullptr;
	Neon::Texture* textureB = nullptr;
	Neon::Texture* textureC = nullptr;

	Neon::RenderData frame("frame");
	Neon::Texture* textureFrame = nullptr;
	Neon::FrameBufferObject* fbo = nullptr;

	app.OnInitialize([&]() {
		auto t = Neon::Time("Initialize");

		auto shaderFixedColor = app.CreateComponent<Neon::Shader>("fixedColor", (app.GetResourceRoot() + "/shader/fixedColor.vs").c_str(), (app.GetResourceRoot() + "/shader/fixedColor.fs").c_str());
		auto shaderTexture = app.CreateComponent<Neon::Shader>("texture", (app.GetResourceRoot() + "/shader/texture.vs").c_str(), (app.GetResourceRoot() + "/shader/texture.fs").c_str());

		shaderA = new Neon::Shader("fixedColor", (app.GetResourceRoot() + "/shader/fixedColor.vs").c_str(), (app.GetResourceRoot() + "/shader/fixedColor.fs").c_str());
		shaderB = new Neon::Shader("texture", (app.GetResourceRoot() + "/shader/texture.vs").c_str(), (app.GetResourceRoot() + "/shader/texture.fs").c_str());


		{
			auto entity = app.CreateEntity("triangleV");

			auto renderData = app.CreateComponent<Neon::RenderData>("triangleV RenderData");
			renderData->AddVertex(-0.125f, 0.0f, 0.0f);
			renderData->AddVertex(0.0f, -0.5f, 0.0f);
			renderData->AddVertex(0.125f, 0.0f, 0.0f);

			renderData->AddIndex(0);
			renderData->AddIndex(1);
			renderData->AddIndex(2);

			renderData->AddShader(shaderA);

			entity->AddComponent(renderData);
		}

		{
			frame.AddVertex(-0.5f, -0.5f, 0.0f);
			frame.AddVertex(0.5f, -0.5f, 0.0f);
			frame.AddVertex(0.5f, 0.5f, 0.0f);
			frame.AddVertex(-0.5f, 0.5f, 0.0f);

			frame.AddIndex(0);
			frame.AddIndex(1);
			frame.AddIndex(2);

			frame.AddIndex(0);
			frame.AddIndex(2);
			frame.AddIndex(3);

			frame.AddUV(0.0f, 0.0f);
			frame.AddUV(1.0f, 0.0f);
			frame.AddUV(1.0f, 1.0f);
			frame.AddUV(0.0f, 1.0f);

			textureFrame = new Neon::Texture("frame", 1920, 1080);
			fbo = new Neon::FrameBufferObject("frame", textureFrame);

			frame.AddTexture(textureFrame);
		}

		{
			auto entity = app.CreateEntity("triangleA");

			auto renderData = app.CreateComponent<Neon::RenderData>("triangleA RenderData");
			renderData->AddVertex(-0.125f, 0.0f, 0.0f);
			renderData->AddVertex(0.125f, 0.0f, 0.0f);
			renderData->AddVertex(0.0f, 0.5f, 0.0f);

			renderData->AddIndex(0);
			renderData->AddIndex(1);
			renderData->AddIndex(2);

			renderData->AddShader(shaderA);

			entity->AddComponent(renderData);
		}

		{
			auto entity = app.CreateEntity("owl");

			auto renderData = app.CreateComponent<Neon::RenderData>("owl RenderData");

			renderData->AddVertex(0.5f, 0.0f, 0.0f);
			renderData->AddVertex(0.75f, 0.0f, 0.0f);
			renderData->AddVertex(0.75f, 0.5f, 0.0f);
			renderData->AddVertex(0.5f, 0.5f, 0.0f);

			renderData->AddIndex(0);
			renderData->AddIndex(1);
			renderData->AddIndex(2);

			renderData->AddIndex(0);
			renderData->AddIndex(2);
			renderData->AddIndex(3);

			renderData->AddUV(0.0f, 0.0f);
			renderData->AddUV(1.0f, 0.0f);
			renderData->AddUV(1.0f, 1.0f);
			renderData->AddUV(0.0f, 1.0f);

			imageB = new Neon::Image("Owl.jpg", app.GetResourceRoot() + "/images/Owl.jpg");
			textureB = new Neon::Texture("Owl", imageB);

			renderData->AddTexture(textureB);
			renderData->AddShader(shaderB);

			entity->AddComponent(renderData);
		}

		{
			auto entity = app.CreateEntity("lion");

			auto renderData = app.CreateComponent<Neon::RenderData>("lion RenderData");

			renderData->AddVertex(-0.5f, 0.0f, 0.0f);
			renderData->AddVertex(-0.25f, 0.0f, 0.0f);
			renderData->AddVertex(-0.25f, 0.5f, 0.0f);
			renderData->AddVertex(-0.5f, 0.5f, 0.0f);

			renderData->AddIndex(0);
			renderData->AddIndex(1);
			renderData->AddIndex(2);

			renderData->AddIndex(0);
			renderData->AddIndex(2);
			renderData->AddIndex(3);

			renderData->AddUV(0.0f, 0.0f);
			renderData->AddUV(1.0f, 0.0f);
			renderData->AddUV(1.0f, 1.0f);
			renderData->AddUV(0.0f, 1.0f);

			imageC = new Neon::Image("Lion", app.GetResourceRoot() + "/images/Lion.png");
			textureC = new Neon::Texture("Lion", imageC);

			renderData->AddTexture(textureC);
			renderData->AddShader(shaderB);

			entity->AddComponent(renderData);
		}

		});





	app.OnUpdate([&](float timeDelta) {
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

		SAFE_DELETE(textureB);
		SAFE_DELETE(textureC);

		SAFE_DELETE(shaderA);
		SAFE_DELETE(shaderB);
		});

	app.Run();

	return 0;
}
