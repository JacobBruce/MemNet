#pragma once
#include <string>
#include <unordered_map>

#define GL_DEBUG        1

#define CONFIG_FILE     "settings.cfg"
#define SUBCHAR_FILE    "char_map.cfg"
#define WINDOW_TITLE	"MemNet"
#define FLOAT_EPSILON   1e-7

namespace GLOBALS {

	inline std::unordered_map<std::string,std::string> config_map;
    inline std::string DATA_FOLDER;
    inline uint32_t WORKGROUP_SIZE;
    inline uint32_t BP_DEPTH;
    inline uint32_t BP_STOP;
    inline float POS_LABEL;
    inline float NEG_LABEL;
}
