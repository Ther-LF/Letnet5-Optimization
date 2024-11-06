CXX = nvcc
CXXFLAGS = -O3 --std=c++17 --gpu-architecture=compute_80 --gpu-code=sm_80,compute_80
INCLUDES = -I ./include
TARGET = inference

all: $(TARGET)

inference: inference.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) 

clean:
	rm -f $(TARGET)