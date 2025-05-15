#pragma once
#include <iostream>
#include <string>
#include <locale>
#include <cstdint>
#include <Windows.h>
#include <parallel_hashmap/phmap.h>
#include "ReadWrite.h"

namespace Worderizer {

    struct SubCharData
    {
        std::u32string sub;
        bool use;

        SubCharData() : use(false) {};
    };

    inline uint32_t MinOccurr = 2;
    inline uint8_t MaxRepLen = 6;
    inline uint8_t MaxNumLen = 4;
    inline uint8_t MaxWordLen = 64;
    inline uint32_t MaxCharCode = 65536;

    inline std::vector<SubCharData> charSubTable;

    inline std::wstring_convert<std::codecvt_utf8<char32_t>,char32_t> cv_u8_u32;
    //std::wstring_convert<std::codecvt_utf16<char32_t>,char32_t> cv_u16_u32;

    inline bool IsUTF8orASCII(const std::string& file_str)
    {
        uint8_t* chars = (uint8_t*)file_str.c_str();

        if ((chars[0] == 0xFF && chars[1] == 0xFE) || (chars[0] == 0xFE && chars[1] == 0xFF)) {
            //std::cout << "File seems to use UTF16 or UTF32" << std::endl;
            return false;
        }

        //std::cout << "File seems to use UTF8 or ASCII" << std::endl;
        return true;
    }

    inline std::string U32ToU8(const std::u32string& str)
    {
        return cv_u8_u32.to_bytes(str);
    }

    inline std::u32string U8ToU32(const std::string& str)
    {
        return cv_u8_u32.from_bytes(str);
    }

    inline void LoadSubChars(const std::string char_file)
    {
        std::unordered_map<std::string,std::string> charMap;

        LoadConfigFile(char_file, charMap);

        charSubTable.reserve(charMap.size());

        for (const auto& n : charMap)
        {
            uint32_t c = stoul(n.first);

            if (c >= charSubTable.size())
                charSubTable.resize(c+1);

            charSubTable[c].sub = U8ToU32(n.second);
            charSubTable[c].use = true;
        }
    }

    inline bool IsInSubTable(uint32_t c)
    {
        if (c >= charSubTable.size()) return false;

        return charSubTable[c].use;
    }

    inline std::u32string NormalizeChars(const std::u32string& str)
    {
        std::u32string result;
        result.reserve(str.size());

        for (const char32_t& c : str)
        {
            if (IsInSubTable(c)) {
                result += charSubTable[c].sub;
            } else {
                result.push_back(c);
            }
        }

        return result;
    }

    inline bool IsAlpha(const uint32_t& c)
    {
        if ((c > 64 && c < 91) || (c > 96 && c < 123)) {
            return true;
        } else if ((c > 191 && c < 448) && (c != 215 && c != 247)) {
            return true;
        } else if ((c > 451 && c < 688)) {
            return true;
        }

        return false;
    }

    inline bool IsDigit(const uint32_t& c)
    {
        return (c > 47 && c < 58);
    }

    inline bool SkipChar(const char32_t& c)
    {
        return (!(iswspace(c) || iswgraph(c)) || iswcntrl(c) || c > MaxCharCode);
    }

    inline bool UpdateWord(std::u32string& word, bool& is_first_char,
                    bool& is_number, bool& next_char, const char32_t& c)
    {
        bool haveWord = false;

        if (is_first_char) {

            if (SkipChar(c)) return false;

            is_first_char = false;
            is_number = false;

            if (IsAlpha(c)) {
                //isAlpha = true;
            } else if (IsDigit(c)) {
                is_number = true;
            } else {
                haveWord = true;
            }

            word.assign(1, c);

        } else {

            if (SkipChar(c)) return true;

            if (is_number) {
                if (IsDigit(c)) {
                    word.push_back(c);
                    if (word.length() >= MaxNumLen)
                        haveWord = true;
                } else {
                    haveWord = true;
                    next_char = true;
                }
            } else {
                if (IsAlpha(c)) {
                    word.push_back(c);
                } else {
                    haveWord = true;
                    next_char = true;
                }
            }
        }

        if (word.length() >= MaxWordLen) haveWord = true;

        return haveWord;
    }

    inline void SetMapIndices(phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words)
    {
        uint32_t cIndex = 0;

        for (auto it = words.begin(); it != words.end();)
        {
            if (it->second >= MinOccurr) {
                it->second = cIndex++;
                it++;
            } else {
                it = words.erase(it);
            }
        }

        if (words.size() > UINT32_MAX)
            HandleFatalError("Word count exceeded UINT32_MAX");
    }

    inline void GenEnglishWordMap(phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words, std::string data_dir, bool set_indices=true)
    {
        std::u32string word, fileText;
        std::string fileStr;
        bool isNumber = false;
        bool nextChar = false;
        bool isFirstChar = true;
        char32_t tempChar = 0;

        for (char32_t i=32; i < 127; ++i)
        {
            word.assign(1, i);
            words[word] = MinOccurr;
        }

        std::vector<std::string> files(ListFiles(data_dir));

        for (const std::string& filePath : files)
        {
            std::cout << "Reading file: " << filePath << std::endl;

            fileStr = ReadFileStr(filePath);

            if (IsUTF8orASCII(fileStr)) {
                fileText = cv_u8_u32.from_bytes(fileStr.data());
            } else {
                continue;
            }

            isFirstChar = true;
            nextChar = false;

            for (const char32_t& c : fileText)
            {
                tempChar = c;

                while(true)
                {
                    if (UpdateWord(word, isFirstChar, isNumber, nextChar, tempChar)) {

                        isFirstChar = true;

                        if (words.contains(word)) {
                            if (words[word] < UINT32_MAX)
                                words[word] += 1;
                        } else {
                            words[word] = 1;
                        }

                        if (nextChar) {
                            nextChar = false;
                            continue;
                        }
                    }

                    break;
                }
            }

            std::cout << "Word Count: " << words.size() << std::endl;
        }

        if (set_indices) SetMapIndices(words);

        //for (const auto& n : words)
            //std::cout << cv_u8_u32.to_bytes(n.first) << ": " << n.second << std::endl;

        std::cout << "Final Word Count: " << words.size() << std::endl;
    }

    inline void GenEnglishWordMapAlt(phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words, std::string data_dir, bool set_indices=true)
    {
        std::u32string word, fileText;
        std::string fileStr;
        bool isNumber = false;
        bool nextChar = false;
        bool isFirstChar = true;
        char32_t tempChar = 0;

        phmap::parallel_flat_hash_map<std::u32string, bool> wordsAlt;

        std::vector<std::string> files(ListFiles(data_dir));

        for (const std::string& filePath : files)
        {
            std::cout << "Reading file: " << filePath << std::endl;

            fileStr = ReadFileStr(filePath);

            if (IsUTF8orASCII(fileStr)) {
                fileText = cv_u8_u32.from_bytes(fileStr.data());
            } else {
                continue;
            }

            wordsAlt.clear();
            isFirstChar = true;
            nextChar = false;

            for (const char32_t& c : fileText)
            {
                tempChar = c;

                while(true)
                {
                    if (UpdateWord(word, isFirstChar, isNumber, nextChar, tempChar)) {

                        isFirstChar = true;

                        if (!wordsAlt.contains(word))
                            wordsAlt[word] = true;

                        if (nextChar) {
                            nextChar = false;
                            continue;
                        }
                    }

                    break;
                }
            }

            for (const auto& n : wordsAlt)
            {
                if (words.contains(n.first)) {
                    if (words[n.first] < UINT32_MAX)
                        words[n.first] += 1;
                } else {
                    words[n.first] = 1;
                }
            }

            std::cout << "Word Count: " << words.size() << std::endl;
        }

        if (set_indices) SetMapIndices(words);

        std::cout << "Final Word Count: " << words.size() << std::endl;
    }

    inline void SaveWordMap(phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words, std::string map_file)
    {
        std::string wordStr, tempStr;
        std::vector<std::u32string> wordVec;
        wordVec.resize(words.size());

        for (const auto& n : words) wordVec[n.second] = n.first;

        FILE* pFile = fopen(map_file.c_str(), "wb");
        if (pFile == NULL) HandleFatalError("Failed to create "+map_file);

        for (const std::u32string& word : wordVec)
        {
            try {
                tempStr.assign(cv_u8_u32.to_bytes(word));
                wordStr.assign(1, tempStr.length());
                wordStr += tempStr;
                fwrite(wordStr.data(), 1, wordStr.size(), pFile);

            } catch (std::range_error& exception) {

                std::cout << "Corrupt word detected. Contains characters: " << word.length() << std::endl;

                for (const auto& c : word) std::cout << "Char code: " << c << std::endl;

                std::string userIn;
                std::cout << "Skip word and continue saving word map? (type Y for yes) ";
                std::cin >> userIn;

                if (StrToUpper(userIn) != "Y") exit(EXIT_FAILURE);
            }
        }

        fclose(pFile);

        std::cout << "Saved word map to " << map_file << std::endl;
    }

    inline void LoadWordMap(phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words, std::string map_file)
    {
        std::u32string word;
        std::string wordStr;
        uint32_t wordIndex = 0;
        uint8_t wordSize = 0;

        words.clear();

        FILE* pFile = fopen(map_file.c_str(), "rb");
        if (pFile == NULL) HandleFatalError("Failed to open "+map_file);

        while (fread(&wordSize, 1, 1, pFile))
        {
            wordStr.resize(wordSize);

            if (fread(wordStr.data(), 1, wordSize, pFile)) {
                word = cv_u8_u32.from_bytes(wordStr.data());
                words[word] = wordIndex++;
            } else {
                HandleFatalError("Corrupt wordmap file detected");
            }
        }

        fclose(pFile);

        std::cout << "Loaded word map with " << words.size() << " tokens" << std::endl;
    }

    inline void DelWordsFromMap(phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words,
                              std::vector<std::u32string> del_words)
    {
        bool foundWord = false;
        uint32_t wordIndex = 0;

        for (auto it = words.begin(); it != words.end();)
        {
            for (const auto& word : del_words)
            {
                if (it->first == word) {
                    it = words.erase(it);
                    foundWord = true;
                    break;
                }
            }

            if (foundWord) {
                foundWord = false;
            } else {
                it++;
            }
        }

        for (auto& n : words) n.second = wordIndex++;
    }

    inline void AddWordsToMap(phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words,
                              std::vector<std::u32string> new_words)
    {
        uint32_t wordIndex = words.size();

        if (wordIndex + new_words.size() > UINT32_MAX)
            HandleFatalError("Size of merged map cannot exceed UINT32_MAX");

        for (const auto& word : new_words)
        {
            if (words.contains(word)) continue;

            words[word] = wordIndex++;

            std::cout << "Added " << U32ToU8(word) << " to word map" << std::endl;
        }

        std::cout << "Final word count " << words.size() << std::endl;
    }

    inline void CleanWordMap(phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words,
                             bool long_clean=true, uint8_t long_len=10, uint8_t min_changes=4)
    {
        std::vector<std::u32string> repWords;

        for (const auto& n : words)
        {
            if (n.first.length() > MaxRepLen) {

                bool isRepWord = true;
                uint32_t firstChar = n.first[0];

                for (size_t c=1; c < n.first.length(); ++c)
                {
                    if (firstChar != n.first[c]) {
                        isRepWord = false;
                        break;
                    }
                }

                if (isRepWord) repWords.push_back(n.first);
            }
        }

        DelWordsFromMap(words, repWords);

        std::cout << "Removed " << repWords.size() << " same-character words" << std::endl;

        if (!long_clean) return;

        repWords.clear();

        for (const auto& n : words)
        {
            if (n.first.length() >= long_len) {

                uint32_t changes = 0;
                uint32_t lastChar = n.first[0];

                for (size_t c=1; c < n.first.length(); ++c)
                {
                    if (lastChar != n.first[c]) {
                        lastChar = n.first[c];
                        changes++;
                    }
                }

                if (changes < min_changes) repWords.push_back(n.first);
            }
        }

        DelWordsFromMap(words, repWords);

        std::cout << "Removed " << repWords.size() << " long repetitive words" << std::endl;
    }

    inline void MergeWordMaps(phmap::parallel_flat_hash_map<std::u32string, uint32_t>& dest_words,
                       const phmap::parallel_flat_hash_map<std::u32string, uint32_t>& more_words)
    {
        uint32_t wordIndex = dest_words.size();

        if (wordIndex + more_words.size() > UINT32_MAX)
            HandleFatalError("Size of merged map cannot exceed UINT32_MAX");

        for (const auto& n : more_words) {
            if (!dest_words.contains(n.first))
                dest_words[n.first] = wordIndex++;
        }
    }

    inline bool StrToTokens(const std::u32string& str, std::vector<uint32_t>& dest,
                     phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words,
                     bool skip_unknowns=true)
    {
        std::u32string word, nextWord, tempStr;
        bool foundToken = false;
        bool isNumber = false;
        bool nextChar = false;
        bool isFirstChar = true;
        char32_t tempChar = 0;

        if (str.empty()) return false;

        std::u32string normStr(NormalizeChars(str));

        for (size_t c=0; c < normStr.length(); ++c)
        {
            tempChar = normStr[c];

            while(true)
            {
                if (UpdateWord(word, isFirstChar, isNumber, nextChar, tempChar)) {

                    isFirstChar = true;
                    foundToken = true;

                    if (!nextChar && word.length() == 1 && c < normStr.length()-1) {

                        tempStr = word;
                        tempStr.push_back(normStr[c+1]);

                        if (words.contains(tempStr)) {
                            dest.push_back(words[tempStr]);
                            c++;
                            break;
                        }
                    }

                    while (!words.contains(word))
                    {
                        nextWord.push_back(word.back());
                        word.pop_back();

                        if (word.length() == 0) {
                            foundToken = false;
                            break;
                        }
                    }

                    if (foundToken) {
                        dest.push_back(words[word]);
                    } else if (!skip_unknowns) {
                        return false;
                    }

                    if (nextWord.length() > 0) {
                        if (nextWord.length() > 1) {
                            word.assign(nextWord.begin(), nextWord.end()-(!foundToken));
                            std::reverse(word.begin(), word.end());
                            isFirstChar = false;
                        } else if (foundToken) {
                            word = nextWord;
                            isFirstChar = false;
                        }

                        nextWord.clear();
                    }

                    if (nextChar) {
                        nextChar = false;
                        continue;
                    }

                }

                break;
            }
        }

        return !dest.empty();
    }

    inline void TokensToStr(std::u32string& dest, const std::vector<uint32_t>& tokens,
                     phmap::parallel_flat_hash_map<std::u32string, uint32_t>& words)
    {
        std::u32string numStr;

        dest.clear();

        for (const uint32_t& token : tokens)
        {
            auto it = std::find_if(words.begin(), words.end(),
                [&token](auto&& n) { return n.second == token; }
            );

            if (it == std::end(words)) {
                HandleFatalError("Unknown token ID: "+std::to_string(token));
            } else {
                dest += it->first;
            }
        }
    }
};
