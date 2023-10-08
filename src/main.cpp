#include <iostream>

#include <Neon/Neon.h>

int main()
{
	Neon::Application app(1920, 1080);
	app.SetResourceRoot(filesystem::current_path().string() + "/../res");

	Neon::Shader* shaderA = nullptr;
	Neon::Shader* shaderB = nullptr;

	Neon::Image* imageB = nullptr;
	Neon::Image* imageC = nullptr;
	Neon::Texture* textureB = nullptr;
	Neon::Texture* textureC = nullptr;

	Neon::RenderData triangle;
	Neon::RenderData owl;
	Neon::RenderData lion;
	
	Neon::RenderData frame;
	Neon::Texture* textureFrame = nullptr;
	Neon::FrameBufferObject* fbo = nullptr;

	app.OnInitialize([&]() {
		auto t = Neon::Time("Initialize");

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
			triangle.AddVertex(-0.125f, 0.0f, 0.0f);
			triangle.AddVertex(0.125f, 0.0f, 0.0f);
			triangle.AddVertex(0.0f, 0.5f, 0.0f);

			triangle.AddIndex(0);
			triangle.AddIndex(1);
			triangle.AddIndex(2);
		}

		{
			owl.AddVertex(0.5f, 0.0f, 0.0f);
			owl.AddVertex(0.75f, 0.0f, 0.0f);
			owl.AddVertex(0.75f, 0.5f, 0.0f);
			owl.AddVertex(0.5f, 0.5f, 0.0f);

			owl.AddIndex(0);
			owl.AddIndex(1);
			owl.AddIndex(2);

			owl.AddIndex(0);
			owl.AddIndex(2);
			owl.AddIndex(3);

			owl.AddUV(0.0f, 0.0f);
			owl.AddUV(1.0f, 0.0f);
			owl.AddUV(1.0f, 1.0f);
			owl.AddUV(0.0f, 1.0f);

			imageB = new Neon::Image("Owl.jpg", app.GetResourceRoot() + "/images/Owl.jpg");
			textureB = new Neon::Texture("Owl", imageB);

			owl.AddTexture(textureB);
		}

		{
			lion.AddVertex(-0.5f, 0.0f, 0.0f);
			lion.AddVertex(-0.25f, 0.0f, 0.0f);
			lion.AddVertex(-0.25f, 0.5f, 0.0f);
			lion.AddVertex(-0.5f, 0.5f, 0.0f);

			lion.AddIndex(0);
			lion.AddIndex(1);
			lion.AddIndex(2);

			lion.AddIndex(0);
			lion.AddIndex(2);
			lion.AddIndex(3);

			lion.AddUV(0.0f, 0.0f);
			lion.AddUV(1.0f, 0.0f);
			lion.AddUV(1.0f, 1.0f);
			lion.AddUV(0.0f, 1.0f);

			imageC = new Neon::Image("Lion", app.GetResourceRoot() + "/images/Lion.png");
			textureC = new Neon::Texture("Lion", imageC);

			lion.AddTexture(textureC);
		}

		shaderA = new Neon::Shader((app.GetResourceRoot() + "/shader/fixedColor.vs").c_str(), (app.GetResourceRoot() + "/shader/fixedColor.fs").c_str());
		shaderB = new Neon::Shader((app.GetResourceRoot() + "/shader/texture.vs").c_str(), (app.GetResourceRoot() + "/shader/texture.fs").c_str());

		});





	app.OnUpdate([&](float timeDelta) {
		//auto t = Neon::Time("Update");

		fbo->Bind();

		glClearColor(0.9f, 0.7f, 0.5f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shaderA->use();
		triangle.Bind();
		glDrawElements(GL_TRIANGLES, (GLsizei)triangle.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		shaderB->use();
		owl.Bind();
		glDrawElements(GL_TRIANGLES, (GLsizei)owl.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		lion.Bind();
		glDrawElements(GL_TRIANGLES, (GLsizei)lion.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		fbo->Unbind();

		glClearColor(0.3f, 0.5f, 0.7f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shaderA->use();
		triangle.Bind();
		glDrawElements(GL_TRIANGLES, (GLsizei)triangle.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		shaderB->use();
		owl.Bind();
		glDrawElements(GL_TRIANGLES, (GLsizei)owl.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		lion.Bind();
		glDrawElements(GL_TRIANGLES, (GLsizei)lion.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);

		frame.Bind();
		glDrawElements(GL_TRIANGLES, (GLsizei)frame.GetIndexBuffer()->Size(), GL_UNSIGNED_INT, 0);
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
