#include <Neon/NeonURL.h>

namespace Neon
{
	string makeRelative(const string& path)
	{
		string p = path;

		if (p.front() == '/') {
			p = p.substr(1);
		}

		return p;
	}

	URL URL::FromRelativePath(const string& relativePath)
	{
		//auto cwd = URL(Settings["Current Working Directory"].get<string>());

		//string rp = relativePath;
		//replace(rp.begin(), rp.end(), '\\', '/');
		//if (rp.front() == '/') {
		//	rp = rp.substr(1);
		//}

		//return cwd + rp;

		return GetCurrentPath() + relativePath;
	}

	URL URL::GetCurrentPath()
	{
		return URL(filesystem::current_path());
		//return URL(Settings["Current Working Directory"].get<string>());
	}

	void URL::SetCurrentPath(const URL& currentPath)
	{
		filesystem::current_path(currentPath.path);
		//Settings["Current Working Directory"] = currentPath;
	}

	URL URL::GetShaderFileURL(const string& shaderFileName)
	{
		auto shaderRootDirectory = URL(Settings["Resource Root Directory"].get<string>()) + string("shader/");
		return URL(shaderRootDirectory + shaderFileName);
	}

	URL URL::GetFontFileURL(const string& fontFileName)
	{
		auto fontRootDirectory = URL(Settings["Resource Root Directory"].get<string>()) + string("fonts/");
		return URL(fontRootDirectory + fontFileName);
	}

	void URL::ChangeDirectory(const string& path)
	{
		URL currentDirectory = URL::GetCurrentPath();
		SetCurrentPath(currentDirectory + path);
	}

	URL URL::Resource(const URL& path)
	{
		return GetCurrentPath() + string("/res/") + path;
	}

	URL::URL()
		: path("")
	{
	}

	URL::URL(const char* path)
		: path(path)
	{
	}

	URL::URL(const string& absolutePath)
		: path(absolutePath)
	{
		replace(path.begin(), path.end(), '\\', '/');
	}

	URL::URL(const filesystem::path& path)
		: path(path.string())
	{
	}

	URL::URL(const URL& other)
		: path(other.path)
	{
	}

	URL::~URL()
	{

	}

	URL URL::operator + (const URL& other)
	{
		if (path.back() == '/')
		{
			return URL(path + makeRelative(other.path));
		}
		else
		{
			return URL(path + "/" + makeRelative(other.path));
		}
	}

	URL& URL::operator += (const URL& other)
	{
		if (path.back() != '/')
		{
			path += "/";
		}
		path += makeRelative(other.path);
		return *this;
	}

	URL URL::operator + (const string& other)
	{
		if (path.back() == '/')
		{
			return URL(path + makeRelative(other));
		}
		else
		{
			return URL(path + "/" + makeRelative(other));
		}
	}

	URL& URL::operator += (const string& other)
	{
		if (path.back() != '/')
		{
			path += "/";
		}
		path += makeRelative(other);
		return *this;
	}

	URL operator + (const URL& a, const URL& b)
	{
		if (a.path.back() == '/')
		{
			return URL(a.path + makeRelative(b.path));
		}
		else
		{
			return URL(a.path + "/" + makeRelative(b.path));
		}
	}

	URL operator + (const URL& a, const string& b)
	{
		if (a.path.back() == '/')
		{
			return URL(a.path + makeRelative(b));
		}
		else
		{
			return URL(a.path + "/" + makeRelative(b));
		}
	}
}