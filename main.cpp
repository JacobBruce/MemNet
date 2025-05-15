#include "Engine.h"

/*
TODO:
- Dropout or random input/output noise
- EarlyStopping (Stop training when net has stopped improving)
- ReduceLROnPlateau (Reduce learning rate when net stops improving)
- Learning rate scheduler (Reduce learning rate over time)
- ModelCheckpoint (Save the model weights at periodic intervals)
- Option to show accuracy metric during training
- Custom source layer(s) for each layer
- Convolutional layers
*/

static void error_callback(int error, const char* description)
{
    std::cerr << "GLFW ERROR (" << error << "): " << description << std::endl;
}

int main(int argc, char *argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    if (argc > 1) {
        GLOBALS::DATA_FOLDER = std::string(argv[1])+"data/";
    } else {
        if (argc == 0) {
            GLOBALS::DATA_FOLDER.assign(__FILE__);
        } else {
            GLOBALS::DATA_FOLDER.assign(argv[0]);
        }

        size_t found = GLOBALS::DATA_FOLDER.find_last_of("/\\");
        GLOBALS::DATA_FOLDER = GLOBALS::DATA_FOLDER.substr(0, found+1)+"data/";
    }

	std::cout << "Loading config file... ";
	LoadConfigFile(GLOBALS::DATA_FOLDER+CONFIG_FILE, GLOBALS::config_map);
    std::cout << "Success!" << std::endl;

    GLOBALS::WORKGROUP_SIZE = stoul(GLOBALS::config_map["WORKGROUP_SIZE"]);

	glfwSetErrorCallback(error_callback);
	std::cout << "Initializing GLFW ... ";

	if (glfwInit() == GLFW_TRUE) {
		std::cout << "Success!" << std::endl;
	} else {
		std::cout << "Failed!" << std::endl;
		exit(EXIT_FAILURE);
	}

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	GLFWwindow* window = glfwCreateWindow(8, 8, WINDOW_TITLE, NULL, NULL);
	glfwMakeContextCurrent(window);

	std::cout << "Initializing GLEW ... ";
	GLenum err = glewInit();

	if (err != GLEW_OK) {
		std::cout << "Failed!\nError: " << glewGetErrorString(err);
		exit(EXIT_FAILURE);
	} else {
		std::cout << "Success!" << std::endl;
	}

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Shader Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

	std::cout << std::endl << "Running engine ..." << std::endl;

    std::vector<std::string> shapeVals(ExplodeStr(GLOBALS::config_map["NET_SHAPE"], ","));
    std::vector<GLuint> netShape;
    netShape.reserve(shapeVals.size());

    if (shapeVals.size() < 3) HandleFatalError("net must have at least one hidden layer");

    for (GLuint i=0; i < shapeVals.size(); ++i)
        netShape.push_back(stoul(shapeVals[i]));

    if (!DirExists(GLOBALS::config_map["NET_DIR"]))
        HandleFatalError("no model folder found at "+GLOBALS::config_map["NET_DIR"]);

    std::string netType(GLOBALS::config_map["NET_TYPE"]);
    std::vector<std::string> actFuncs(ExplodeStr(GLOBALS::config_map["ACT_FUNCS"], ","));
    std::vector<std::string> memLayers(ExplodeStr(GLOBALS::config_map["MEM_LAYERS"], ","));
    std::string memBlendFunc(GLOBALS::config_map["MEM_BLEND_FUNC"]);
    GLfloat minWeight = stof(GLOBALS::config_map["MIN_WEIGHT"]);
    GLfloat maxWeight = stof(GLOBALS::config_map["MAX_WEIGHT"]);
    GLfloat initBias = stof(GLOBALS::config_map["INIT_BIAS"]);
    GLfloat initFrate = stof(GLOBALS::config_map["INIT_FRATE"]);
    GLuint trainEpochs = stoul(GLOBALS::config_map["TRAIN_EPOCHS"]);
    GLuint examplesPB = stoul(GLOBALS::config_map["EXAMPLES_PB"]);
    GLuint trainSteps = stoul(GLOBALS::config_map["TRAIN_STEPS"]);
    trainSteps = (trainSteps < 2 ? 1 : trainSteps);
    float4 initData = { minWeight, maxWeight, initBias, initFrate };

    Engine engine(trainSteps > 1);

	if (GLOBALS::config_map["ENGINE_MODE"] == "0") { // GEN NET

        engine.GenNet(netShape, netType, initData, actFuncs, memLayers, memBlendFunc);

        engine.SaveNet(GLOBALS::config_map["NET_DIR"], GLOBALS::config_map["NET_NAME"]);

    } else if (GLOBALS::config_map["ENGINE_MODE"] == "1") { // TRAIN NET

        if (FileExists(GLOBALS::config_map["NET_DIR"]+GLOBALS::config_map["NET_NAME"]+".config")) {
            engine.LoadNet(GLOBALS::config_map["NET_DIR"], GLOBALS::config_map["NET_NAME"]);
        } else {
            HandleFatalError("no net named "+GLOBALS::config_map["NET_NAME"]+" could be found");
        }

        engine.LoadDataset(
            GLOBALS::config_map["TRAIN_DATA"], GLOBALS::config_map["DATA_TYPE"],
            GLOBALS::config_map["INPUT_FUNC"], GLOBALS::config_map["OUTPUT_FUNC"],
            examplesPB
        );

        if (StrToUpper(netType) == "WORD_2_VEC") {
            engine.TrainWord2Vec(trainEpochs, trainSteps);
        } else if (StrStartsWith(netType, "AUTO_ENC")) {
            std::cout << "Training auto-encoder ..." << std::endl;
            if (StrToUpper(GLOBALS::config_map["INPUT_FUNC"]) == "TOKENIZE") {
                engine.TrainTextAutoEncoder(trainEpochs, trainSteps);
            } else {
                engine.TrainAutoEncoder(trainEpochs, trainSteps);
            }
        } else if (StrToUpper(netType) == "TEXT_GEN") {
            std::cout << "Training text model ..." << std::endl;
            engine.TrainTextNet(trainEpochs, trainSteps);
        } else {
            std::cout << "Training model ..." << std::endl;
            engine.TrainNet(trainEpochs, trainSteps);
        }

        engine.SaveNet(GLOBALS::config_map["NET_DIR"], GLOBALS::config_map["NET_NAME"]);

    } else if (GLOBALS::config_map["ENGINE_MODE"] == "2") { // TEST NET

        engine.LoadNet(GLOBALS::config_map["NET_DIR"], GLOBALS::config_map["NET_NAME"]);

        engine.LoadDataset(
            GLOBALS::config_map["TEST_DATA"], GLOBALS::config_map["DATA_TYPE"],
            GLOBALS::config_map["INPUT_FUNC"], GLOBALS::config_map["OUTPUT_FUNC"],
            examplesPB
        );

        engine.TestNet();

    } else { // RUN NET

        engine.LoadNet(GLOBALS::config_map["NET_DIR"], GLOBALS::config_map["NET_NAME"]);

        engine.RunNet();
    }

	std::cout << std::endl << "Stopping engine ..." << std::endl;

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
