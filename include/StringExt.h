#pragma once
#include <cstdlib>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>
#include <cctype>

inline void ReplaceStr(std::string& s, std::string search, std::string replace)
{
    for (size_t pos = 0; ; pos += replace.length()) {
        pos = s.find(search, pos);
		if (pos == std::string::npos) { break; };
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
}

inline std::string ReplaceStr(const std::string& s, std::string search, std::string replace)
{
    std::string result(s);

    for (size_t pos = 0; ; pos += replace.length()) {
        pos = result.find(search, pos);
		if (pos == std::string::npos) { break; };
        result.erase(pos, search.length());
        result.insert(pos, replace);
    }

    return result;
}

inline std::size_t CountSubStr(const std::string& text, const std::string& str)
{
    std::size_t pos = 0;
    std::size_t occurrences = 0;

    while ((pos = text.find(str, pos)) != std::string::npos) {
        occurrences++;
        pos += str.length();
    }

    return occurrences;
}

inline std::vector<std::string> ExplodeStr(const std::string& text, const std::string sep)
{
    std::vector<std::string> tokens;
    std::size_t start = 0, end = 0;

    while ((end = text.find(sep, start)) != std::string::npos) {
        if (end != start) tokens.emplace_back(text.substr(start, end - start));
        start = end + sep.length();
    }

    if (end != start) {
        if (tokens.empty()) {
            tokens.emplace_back(text);
        } else {
            tokens.emplace_back(text.substr(start));
        }
    }

    return tokens;
}

inline void SplitStr(const std::string& text, const std::string& sep,
                                        std::vector<std::string>& dest)
{
    std::size_t start = 0, end = 0;

    while ((end = text.find(sep, start)) != std::string::npos) {
        if (end != start) dest.emplace_back(text.substr(start, end - start));
        start = end + sep.length();
    }

    if (end != start) {
        if (dest.empty()) {
            dest.emplace_back(text);
        } else {
            dest.emplace_back(text.substr(start));
        }
    }
}

inline std::string& TrimL(std::string &s, const char c)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](int ch) {
        return c != ch;
    }));

    return s;
}

inline std::string TrimL(const std::string &s, const char c)
{
	std::string result(s);
	
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [&](int ch) {
        return c != ch;
    }));

    return result;
}

inline std::string& TrimR(std::string &s, const char c)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](int ch) {
        return c != ch;
    }).base(), s.end());

    return s;
}

inline std::string TrimR(const std::string &s, const char c)
{
	std::string result(s);
	
    result.erase(std::find_if(result.rbegin(), result.rend(), [&](int ch) {
        return c != ch;
    }).base(), result.end());

    return result;
}

inline std::string& TrimStr(std::string &s, const std::string &chars)
{
    for (const char& c : chars) TrimR(TrimL(s, chars[c]), chars[c]);

    return s;
}

inline std::string TrimStr(const std::string &s, const std::string &chars)
{
	std::string result(s);
	
    for (const char& c : chars) TrimR(TrimL(result, chars[c]), chars[c]);

    return result;
}

inline std::string& StrToUpper(std::string& str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::toupper);
    return str;
}

inline std::string StrToUpper(const std::string& str)
{
	std::string result(str);
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

inline std::string& StrToLower(std::string& str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

inline std::string StrToLower(const std::string& str)
{
	std::string result(str);
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

inline bool StrStartsWith(const std::string& str, const std::string& start)
{
    if (start.size() > str.size()) return false;
    return std::mismatch(start.begin(), start.end(), str.begin()).first == start.end();
}

inline bool StrEndsWith(const std::string& str, const std::string& end)
{
    if (end.size() > str.size()) return false;
    return std::equal(end.rbegin(), end.rend(), str.rbegin());
}

inline bool StrIsFalse(const std::string& str)
{
	return (str.empty() || str == "0" || StrToUpper(str) == "FALSE");
}

inline bool StrIsTrue(const std::string& str)
{
	return !StrIsFalse(str);
}

inline std::string RandomDigits(const int len)
{
    static const std::string digits1("123456789");
    static const std::string digitsN("0123456789");

    std::string result;
    result.reserve(len);

    result = digits1[rand() % (digits1.length() - 1)];

    for (int i=1; i < len; ++i)
        result += digitsN[rand() % (digitsN.length() - 1)];

    return result;
}
