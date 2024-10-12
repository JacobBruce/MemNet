#pragma once
#include <cstdlib>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>
#include <cctype>

inline void ReplaceSubstr(std::string& s, std::string search, std::string replace)
{
    for (size_t pos = 0; ; pos += replace.length()) {
        pos = s.find(search, pos);
		if (pos == std::string::npos) { break; };
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
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

inline void SplitText(const std::string& text, const std::string& sep,
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

inline std::string& TrimR(std::string &s, const char c)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](int ch) {
        return c != ch;
    }).base(), s.end());

    return s;
}

inline std::string& TrimStr(std::string &s, const std::string &chars)
{
    for (const char& c : s) TrimR(TrimL(s, chars[c]), chars[c]);

    return s;
}

inline std::string& StrToUpper(std::string& str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::toupper);
    return str;
}

inline std::string& StrToLower(std::string& str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

inline std::u32string RandomDigits(const int len)
{
    static const std::u32string digits1(U"123456789");
    static const std::u32string digitsN(U"0123456789");

    std::u32string result;
    result.reserve(len);

    result = digits1[rand() % (digits1.length() - 1)];

    for (int i=1; i < len; ++i)
        result += digitsN[rand() % (digitsN.length() - 1)];

    return result;
}
