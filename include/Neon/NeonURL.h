#pragma once

#include <Neon/NeonCommon.h>
#include <Neon/NeonURL.h>

namespace Neon
{
	class URL
	{
	public:
		static URL FromRelativePath(const string& relativePath);
		static URL GetCurrentPath();
		static void SetCurrentPath(const URL& currentPath);
		static URL GetShaderFileURL(const string& shaderFileName);
		static URL GetFontFileURL(const string& fontFileName);

		static void ChangeDirectory(const string& path);

		static URL Resource(const URL& path);

		URL();
		URL(const filesystem::path& path);
		URL(const char* absolutePath);
		URL(const string& absolutePath);
		URL(const URL& other);
		~URL();

		URL operator + (const URL& other);
		URL& operator += (const URL& other);

		URL operator + (const string& other);
		URL& operator += (const string& other);

		string path;
	};

	URL operator + (const URL& a, const URL& b);
	URL operator + (const URL& a, const string& b);
}
