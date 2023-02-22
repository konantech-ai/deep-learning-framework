#pragma once

#include <time.h>

#include <map>
#include <list>
#include <mutex>
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <set>
#include <random>
#include <condition_variable>
#include <cstring>
using namespace std;

#ifdef FOR_LINUX
typedef int64_t int64;
#else
typedef long long int int64;
#endif
typedef int VRetCode;

enum class VDataType { float32, int32, int64, uint8, bool8, float64 };
enum class VValueType { none, int32, int64, float32, pint32, pint64, pfloat, kbool, string, list, dict, map, shape, object };

extern int VDataTypeSize(VDataType type);
extern string VDataTypeName(VDataType type);

enum class VObjType {
	value, list, dict, map, shape, exception,
	Session, Module, Loss, Metric, Tensor, Parameters, Optimizer, Function,
	DeviceManager, TensorData, Graph, GraphNode, ExecTracer, ExecTracerPool, HyperManager, CbItem, CbBackInfo, CbBackSlot, UDFItem,
	custom };

enum class ActFunc {
	none, relu, leaky, sigmoid, tanh, gelu, selu, mish, swish, softmax
};

class VSession;
class VDict;

class VObjCore;

class VModule;
class VTensor;
class VParameters;
class VLoss;
class VMetric;
class VOptimizer;
class VFunction;

typedef int64 VHandle;

typedef VHandle VHSession;
typedef VHandle VHModule;
typedef VHandle VHTensor;
typedef VHandle VHData;
typedef VHandle VHParameters;
typedef VHandle VHLoss;
typedef VHandle VHMetric;
typedef VHandle VHOptimizer;
typedef VHandle VHFunction;

typedef vector<int> VIntList;
typedef vector<std::string> VStrList;
