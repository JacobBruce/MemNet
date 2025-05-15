#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <assert.h>
#include <sys/stat.h>
#include "StringExt.h"

#ifdef _WIN32
    #include <io.h>
    #include <direct.h>
    #define access   _access_s
    #define stat64   _stat64
    #define mkdir    _mkdir
#else
    #include <unistd.h>
#endif

inline bool DirExists(const std::string& dirname)
{
    struct stat info;
    if (stat(dirname.c_str(), &info) != 0) {
        return false;
    } else if (info.st_mode & S_IFDIR) {
        return true;
    } else {
        return false;
    }
}

inline bool CreateDir(const std::string& dirname)
{
    return (mkdir(dirname.c_str()) == 0) ? true : false;
}

inline bool FileExists(const std::string& filename)
{
    return access(filename.c_str(), 0) == 0;
}

inline long long FileSize(const std::string& filename)
{
    struct stat64 stat_buf;
    int rc = stat64(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

inline void HandleFatalError(std::string emsg)
{
	std::cerr << "ERROR: " << emsg << std::endl;
	exit(EXIT_FAILURE);
}

inline std::vector<std::string> ListFolders(std::string dirname)
{
    std::vector<std::string> result;
    for (const auto& p : std::filesystem::directory_iterator(dirname))
        if (p.is_directory()) result.push_back(p.path().string());
    return result;
}

inline std::vector<std::string> ListFiles(std::string dirname)
{
    std::vector<std::string> result;

    for (const auto& entry : std::filesystem::directory_iterator(dirname))
        if (!entry.is_directory()) result.push_back(entry.path().string());

    return result;
}

inline std::vector<std::string> ListAllFiles(std::string dirname)
{
    std::vector<std::string> result;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(dirname))
        if (!entry.is_directory()) result.push_back(entry.path().string());

    return result;
}

inline std::string ReadFileStr(const std::string filename)
{
    std::string fileStr;

	try {
        std::ifstream sourceFile(filename);
        fileStr.assign(std::istreambuf_iterator<char>(sourceFile),
                              std::istreambuf_iterator<char>());
	} catch(...) {
        HandleFatalError("Couldn't read file: " + filename);
	}

    return fileStr;
}

inline void WriteFileStr(const std::string filename, const std::string str, bool append=false)
{
	std::ofstream destFile;
	
	if (append) {
		destFile.open(filename, std::ios_base::app);
	} else {
		destFile.open(filename);
	}
	
    destFile << str;
	destFile.close();
}

inline std::vector<std::string> ReadFileLines(const std::string filename)
{
    std::ifstream ifs(filename);
    std::vector<std::string> result;

	try {
        if (ifs.is_open()) {
            while (!ifs.eof()) {
                std::string line;
                std::getline(ifs, line);
                result.push_back(line);
            }

            ifs.close();
        } else {
            HandleFatalError("Couldn't open file: " + filename);
        }
	} catch(...) {
        HandleFatalError("Couldn't read file: " + filename);
	}

	return result;
}

inline void LoadConfigFile(std::string filename, std::unordered_map<std::string,std::string>& str_map)
{
	std::ifstream configFile(filename);
	std::string line, key, data;
	size_t bpos;

	try {
        if (configFile.is_open()) {

            while (!configFile.eof()) {
                std::getline(configFile, line);
                if (line.empty() || line[0] == '#') continue;
                bpos = line.find("=");
                if (bpos == std::string::npos) continue;
                key = line.substr(0, bpos);
                data = line.substr(bpos+1);
                str_map[key] = data;
            }

            configFile.close();
            return;

        } else {
            HandleFatalError("Couldn't open config file: " + filename);
        }
	} catch(...) {
        HandleFatalError("Couldn't read config file: " + filename);
	}
}

inline size_t CountStrInFile(const std::string& filename, const std::string& str)
{
    std::ifstream fileStream(filename);
	std::string word;
	uint32_t result = 0;

	try {
        if (fileStream.is_open()) {
            while (fileStream >> word)
                if (word == str) result++;
        } else {
            HandleFatalError("Couldn't open file: " + filename);
        }
	} catch(...) {
        HandleFatalError("Couldn't read file: " + filename);
	}

	return result;
}

inline void SplitTextFile(const std::string& filename, const std::string& sep,
                          std::vector<std::string>& dest, size_t min_chunk_len=2)
{
	std::string word(sep.length(), 0);
	std::string textChunk;
	size_t startIndex = 0;
	size_t endIndex = 0;

    FILE* pFile = nullptr;
    errno_t err = fopen_s(&pFile, filename.c_str(), "rb");
    if (err != 0 || pFile == nullptr) HandleFatalError("failed to open "+filename);

    while (fread(word.data(), 1, sep.size(), pFile) == sep.length())
    {
        if (word == sep) {

            if (endIndex > startIndex) {
                fseek(pFile, startIndex, SEEK_SET);
                textChunk.resize(endIndex-startIndex);
                fread(textChunk.data(), 1, textChunk.size(), pFile);
                if (textChunk.length() >= min_chunk_len) dest.emplace_back(textChunk);
                startIndex = endIndex + sep.length();
                endIndex = startIndex;
            } else {
                endIndex++;
                fseek(pFile, endIndex, SEEK_SET);
            }

        } else {
            endIndex++;
            fseek(pFile, endIndex, SEEK_SET);
        }
    }

    if (dest.empty()) {
        dest.emplace_back(ReadFileStr(filename));
    } else if (endIndex > startIndex) {
        size_t chunkLen = FileSize(filename) - startIndex;
        if (chunkLen > min_chunk_len) {
            fseek(pFile, startIndex, SEEK_SET);
            textChunk.resize(chunkLen);
            fread(textChunk.data(), 1, textChunk.size(), pFile);
            dest.emplace_back(textChunk);
        }
    }

    fclose(pFile);
}
