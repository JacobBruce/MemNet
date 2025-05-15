# MemNet

MemNet is an experimental framework for running and training AI with compute shaders. OpenGL compute shaders are used for their speed and portability, allowing MemNet AI models to run on a wide range of hardware.

The name MemNet comes from the experimental neuron architecture which allows neurons to memorize their previous outputs and has a learnable parameter which determines how quickly those "memories" are forgotten.

The idea is to have a model where the internal state of the network holds information about the past, and that has an impact on future outputs, unlike a tradional neural network where the internal state is constantly reset.

Each neuron has a "memory cell" which should allow the network to behave like a recurrent network but still be computed with a simple forward pass. The memory feature can easily be disabled to create a more traditional feed forward network.

## Getting Started

Download the latest release of MemNet then edit the settings.cfg file inside the data folder. It contains comments to explain what each setting does. First set ENGINE_MODE to 0 to generate a new model and then set ENGINE_MODE to 1 to train the model on some data.

The included settings.cfg file is configured to work with the [MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) in CSV format. You will just need to ensure TRAIN_DATA and TEST_DATA are set to the correct locations, and also change NET_DIR to wherever you want the model files to be saved.

**NOTE:** This project is still highly experimental and in early development so some features may not exist or may not work as expected.

## Dependencies

- [Worderizer](https://github.com/JacobBruce/Worderizer) (already included)
- [parallel-hashmap](https://github.com/greg7mdp/parallel-hashmap) (copy parallel_hashmap folder into the includes folder)
- [AudioFile](https://github.com/adamstark/AudioFile) (copy AudioFile.h into the includes folder)
- [GLFW3](https://www.glfw.org/)
- GLEW
- OpenGL 4.6