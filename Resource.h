#pragma once
#include <string>
#include <unordered_map>

#define GL_DEBUG        1

#define CONFIG_FILE     "settings.cfg"
#define WINDOW_TITLE	"MemNet"

namespace GLOBALS {

	inline std::unordered_map<std::string,std::string> config_map;
    inline std::string DATA_FOLDER;
    inline uint32_t WORKGROUP_SIZE;
}
