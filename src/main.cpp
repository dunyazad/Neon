#include <iostream>

#include <Neon/Neon.h>

int main()
{
	Neon::Shader* shaderA = nullptr;
	Neon::Shader* shaderB = nullptr;

	Neon::VertexArrayObject* pVAOA = nullptr;
	Neon::VertexArrayObject* pVAOB = nullptr;
	Neon::VertexArrayObject* pVAOC = nullptr;

	Neon::VertexBufferObject<float>* pVBOA = nullptr;
	Neon::VertexBufferObject<float>* pVBOB = nullptr;
	Neon::VertexBufferObject<float>* pVBOC = nullptr;
	Neon::VertexBufferObject<GLuint>* pIBOB = nullptr;
	Neon::VertexBufferObject<GLuint>* pIBOC = nullptr;
	Neon::VertexBufferObject<float>* pUVBOB = nullptr;
	Neon::VertexBufferObject<float>* pUVBOC = nullptr;

	Neon::Image* imageB = nullptr;
	Neon::Image* imageC = nullptr;
	Neon::Texture* textureB = nullptr;
	Neon::Texture* textureC = nullptr;


	Neon::Application app(1920, 1080);
	app.SetResourceRoot(filesystem::current_path().string() + "/../res");

	app.OnInitialize([&]() {
		auto t = Neon::Time("Initialize");

		{
			pVAOA = new Neon::VertexArrayObject();
			pVAOA->Bind();

			pVBOA = new Neon::VertexBufferObject<float>(Neon::VertexBufferObject<float>::VERTEX_BUFFER, 0);
			pVBOA->AddElement(-0.125f);
			pVBOA->AddElement(0.0f);
			pVBOA->AddElement(0.0f);

			pVBOA->AddElement(0.125f);
			pVBOA->AddElement(0.0f);
			pVBOA->AddElement(0.0f);

			pVBOA->AddElement(0.0f);
			pVBOA->AddElement(0.5f);
			pVBOA->AddElement(0.0f);
		}

		{
			pVAOB = new Neon::VertexArrayObject();
			pVAOB->Bind();

			pVBOB = new Neon::VertexBufferObject<float>(Neon::VertexBufferObject<float>::VERTEX_BUFFER, 0);
			pVBOB->AddElement(0.5f);
			pVBOB->AddElement(0.0f);
			pVBOB->AddElement(0.0f);

			pVBOB->AddElement(0.75f);
			pVBOB->AddElement(0.0f);
			pVBOB->AddElement(0.0f);

			pVBOB->AddElement(0.75f);
			pVBOB->AddElement(0.5f);
			pVBOB->AddElement(0.0f);

			pVBOB->AddElement(0.5f);
			pVBOB->AddElement(0.5f);
			pVBOB->AddElement(0.0f);

			pIBOB = new Neon::VertexBufferObject<GLuint>(Neon::VertexBufferObject<GLuint>::INDEX_BUFFER, 0);
			pIBOB->AddElement(0);
			pIBOB->AddElement(1);
			pIBOB->AddElement(2);

			pIBOB->AddElement(0);
			pIBOB->AddElement(2);
			pIBOB->AddElement(3);

			pUVBOB = new Neon::VertexBufferObject<float>(Neon::VertexBufferObject<float>::UV_BUFFER, 1);
			pUVBOB->AddElement(0.0f);
			pUVBOB->AddElement(0.0f);

			pUVBOB->AddElement(1.0f);
			pUVBOB->AddElement(0.0f);

			pUVBOB->AddElement(1.0f);
			pUVBOB->AddElement(1.0f);

			pUVBOB->AddElement(0.0f);
			pUVBOB->AddElement(1.0f);
		}

		{
			pVAOC = new Neon::VertexArrayObject();
			pVAOC->Bind();

			pVBOC = new Neon::VertexBufferObject<float>(Neon::VertexBufferObject<float>::VERTEX_BUFFER, 0);
			pVBOC->AddElement(-0.5f);
			pVBOC->AddElement(0.0f);
			pVBOC->AddElement(0.0f);

			pVBOC->AddElement(-0.25f);
			pVBOC->AddElement(0.0f);
			pVBOC->AddElement(0.0f);

			pVBOC->AddElement(-0.25f);
			pVBOC->AddElement(0.5f);
			pVBOC->AddElement(0.0f);

			pVBOC->AddElement(-0.5f);
			pVBOC->AddElement(0.5f);
			pVBOC->AddElement(0.0f);

			pIBOC = new Neon::VertexBufferObject<GLuint>(Neon::VertexBufferObject<GLuint>::INDEX_BUFFER, 0);
			pIBOC->AddElement(0);
			pIBOC->AddElement(1);
			pIBOC->AddElement(2);

			pIBOC->AddElement(0);
			pIBOC->AddElement(2);
			pIBOC->AddElement(3);

			pUVBOC = new Neon::VertexBufferObject<float>(Neon::VertexBufferObject<float>::UV_BUFFER, 1);
			pUVBOC->AddElement(0.0f);
			pUVBOC->AddElement(0.0f);

			pUVBOC->AddElement(1.0f);
			pUVBOC->AddElement(0.0f);

			pUVBOC->AddElement(1.0f);
			pUVBOC->AddElement(1.0f);

			pUVBOC->AddElement(0.0f);
			pUVBOC->AddElement(1.0f);
		}

		imageB = new Neon::Image("Owl.jpg", app.GetResourceRoot() + "/images/Owl.jpg");
		textureB = new Neon::Texture("Owl", imageB);

		imageC = new Neon::Image("Lion", app.GetResourceRoot() + "/images/Lion.png");
		textureC = new Neon::Texture("Lion", imageC);

		shaderA = new Neon::Shader((app.GetResourceRoot() + "/shader/fixedColor.vs").c_str(), (app.GetResourceRoot() + "/shader/fixedColor.fs").c_str());
		shaderB = new Neon::Shader((app.GetResourceRoot() + "/shader/texture.vs").c_str(), (app.GetResourceRoot() + "/shader/texture.fs").c_str());

		});





	app.OnUpdate([&](float timeDelta) {
		//auto t = Neon::Time("Update");

		glClearColor(0.3f, 0.5f, 0.7f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shaderA->use();
		pVAOA->Bind();
		pVBOA->Bind();
		glDrawArrays(GL_TRIANGLES, 0, (GLsizei)pVBOA->GetElements().size());

		shaderB->use();
		pVAOB->Bind();
		pVBOB->Bind();
		pUVBOB->Bind();
		pIBOB->Bind();
		textureB->Bind();
		glDrawElements(GL_TRIANGLES, (GLsizei)pIBOB->Size(), GL_UNSIGNED_INT, 0);

		pVAOC->Bind();
		pVBOC->Bind();
		pUVBOC->Bind();
		pIBOC->Bind();
		textureC->Bind();
		glDrawElements(GL_TRIANGLES, (GLsizei)pIBOC->Size(), GL_UNSIGNED_INT, 0);
		});






	app.OnTerminate([&]() {
		auto t = Neon::Time("Terminate");

		SAFE_DELETE(pVBOA);
		SAFE_DELETE(pVAOA);
		
		SAFE_DELETE(imageB);
		SAFE_DELETE(imageC);

		SAFE_DELETE(textureB);
		SAFE_DELETE(textureC);

		SAFE_DELETE(pUVBOB);
		SAFE_DELETE(pUVBOC);
		SAFE_DELETE(pIBOB);
		SAFE_DELETE(pIBOC);
		SAFE_DELETE(pVBOB);
		SAFE_DELETE(pVBOC);
		SAFE_DELETE(pVAOB);
		SAFE_DELETE(pVAOC);

		SAFE_DELETE(shaderA);
		SAFE_DELETE(shaderB);
		});

	app.Run();

	return 0;
}
