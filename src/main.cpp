#include <iostream>

#include <Neon/Neon.h>

int main()
{
	Neon::Shader* shader = nullptr;

	unsigned int VBO, VAO;

	Neon::VertexArrayObject* pVAO = nullptr;
	Neon::VertexBufferObject<float>* pVBOA = nullptr;

	Neon::Application app(1920, 1080);
	app.SetResourceRoot(filesystem::current_path().string() + "/../res");

	app.OnInitialize([&]() {
		auto t = Neon::Time("Initialize");

		pVAO = new Neon::VertexArrayObject();

		pVBOA = new Neon::VertexBufferObject<float>(Neon::VertexBufferObject<float>::VERTEX_BUFFER, 0);
		pVBOA->AddElement(0.0f);
		pVBOA->AddElement(0.5f);
		pVBOA->AddElement(0.0f);

		pVBOA->AddElement(0.5f);
		pVBOA->AddElement(-0.5f);
		pVBOA->AddElement(0.0f);


		pVBOA->AddElement(-0.5f);
		pVBOA->AddElement(-0.5f);
		pVBOA->AddElement(0.0f);

		//{
		//	// Define the vertices of the triangle
		//	float vertices[] = {
		//		 0.0f,  0.5f, 0.0f,
		//		 0.5f, -0.5f, 0.0f,
		//		-0.5f, -0.5f, 0.0f
		//	};

		//	// Create the vertex buffer object and vertex array object
		//	glGenBuffers(1, &VBO);
		//	glGenVertexArrays(1, &VAO);

		//	// Bind the vertex array object and vertex buffer object
		//	glBindVertexArray(VAO);
		//	glBindBuffer(GL_ARRAY_BUFFER, VBO);

		//	// Copy the vertices data into the vertex buffer object
		//	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		//	// Specify the vertex attribute pointers
		//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		//	glEnableVertexAttribArray(0);

		//	//// Create the shader program
		//	shader = new Neon::Shader((app.GetResourceRoot() + "/shader/vertexShader.glsl").c_str(), (app.GetResourceRoot() + "/shader/fragmentShader.glsl").c_str());
		//}

		shader = new Neon::Shader((app.GetResourceRoot() + "/shader/vertexShader.glsl").c_str(), (app.GetResourceRoot() + "/shader/fragmentShader.glsl").c_str());

		});





	app.OnUpdate([&](float timeDelta) {
		//auto t = Neon::Time("Update");

		// Clear the screen
		glClearColor(0.3f, 0.5f, 0.7f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//// Draw the triangle
		shader->use();
		pVAO->Bind();
		//glBindVertexArray(VAO);
		pVBOA->Bind();
		glDrawArrays(GL_TRIANGLES, 0, 3);
		});






	app.OnTerminate([&]() {
		auto t = Neon::Time("Terminate");

		SAFE_DELETE(pVBOA);

		SAFE_DELETE(pVAO);

		SAFE_DELETE(shader);

		// Cleanup
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		});

	app.Run();

	return 0;
}
