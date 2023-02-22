#pragma once

#include "../objects/tp_loss.h"
#include "../objects/tp_function.h"
#include "../utils/tp_common.h"

class ApiConn;
class ENNCore;
class TpStreamIn;
class EOptimizer;
class EParameters;

class ENN {
public:
    ENN();
    ENN(const ENN& src);
    ENN(ENNCore* core);
    virtual ~ENN();
    ENN& operator =(const ENN& src);
    bool isValid();
    void close();
    ENN nn() { return *this; }
    ENNCore* getCore() { return m_core; }
    ENNCore* createApiClone();
protected:
    ENNCore* m_core;

public:
    ENN(string server_url, string client_url = "");

public: // for Database access  // not exported
    void login(string username, string password);
    void logout();
    void registrate(string username, string password, string email);
    VList getUserList();
    VDict getUserInfo(string username = ""); // username = "" for me, nonzero for others(for administrator only)
    void setUserInfo(VDict userInfo, string username = ""); // username = "" for me, nonzero for others(for administrator only)
    void closeAccount();
    void removeUser(string username);

    VList getRoles();
    VList getUserRoles(string username = "");
    VList getRolePermissions(string rolename);
    VList getUserPermissions(string username = "");
    void addRole(string rolename);
    void remRole(string rolename, bool force = false);
    void addUserRole(string username, string rolename);
    void remUserRole(string username, string rolename);
    void addRolePermission(string rolename, string permission);
    void remRolePermission(string rolename, string permission);

    void registModel(EModule model, string name, string desc, bool is_public);
    VList getModelList();
    EModule fetchModel(int mid);
    EModule fetchModel(string name);

public:
    int getEngineObjId(EcoObjCore* pCore);

public:
    ApiConn* getApiConn(); // not exported

    void registTensorHandle(VHTensor hTensor); // not exported

    void srand(int64 random_seed);

    void set_no_grad();
    void unset_no_grad();

    void set_no_tracer();
    void unset_no_tracer();

    EModule createModule(string sBuiltin, VDict kwArgs = {}, EModule* pModule = NULL); // not exported

    void saveModule(EModule module, string filename);

    EModule loadModule(VDict moduleInfo); // not exported
    EModule loadModule(string filename);
    EModule loadModule(string root, string filename); // not exported

    void registModuleMap(VHModule hModule, EModule* pModule); // not exported

    int addForwardCallbackHandler(TCbForwardCallback* pCbFunc, VDict instInfo, VDict filters); // not exported
    int addBackwardCallbackHandler(TCbBackwardCallback* pCbFunc, VDict instInfo, VDict filters); // not exported

    void removeCallbackHandler(int nId); // not exported

    VStrList GetLayerNames();
    string GetLayerFormula(string layerName);

    void registUserDefFunc(VHFunction hFunction, EFunction* pUserFunc); // not exported

    ETensor funcCbForward(VHFunction hFunction, int nInst, ETensorList operands, VDict opArgs); // not exported
    ETensor funcCbBackward(VHFunction hFunction, int nInst, ETensor ygrad, int nth, ETensorList operands, VDict opArgs); // not exported

public:
    string get_engine_version();

    bool Cuda_isAvailable();
    int Cuda_getDeviceCount();

    VDict get_builtin_names();
    VStrList get_builtin_names(string domain);
    bool isInBuiltinName(string domain, string sBuiltin);

    VDict getLeakInfo(bool sessionOnly);
    void dumpLeakInfo(bool sessionOnly);

public:
    EModule createUserDefinedLayer(string name, string formula, VDict paramInfo, VDict kwArgs = {});

    EModule Model(string modelName, VDict kwArgs);

    EModule Macro(string macroName, VDict kwArgs = {});

    void RegistMacro(string macroName, EModule contents, VDict kwArgs = {});

    EModule Linear(VDict kwArgs);
    EModule Dense(VDict kwArgs);
    EModule Conv(VDict kwArgs);
    EModule Conv1D(VDict kwArgs);
    EModule Conv2D(VDict kwArgs);
    EModule Conv2D_Transposed(VDict kwArgs);
    EModule Conv2D_Dilated(VDict kwArgs);
    EModule Conv2D_Separable(VDict kwArgs);  // 미구현 상태
    EModule Conv2D_Depthwise_Separable(VDict kwArgs);  // 미구현 상태
    EModule Conv2D_Pointwise(VDict kwArgs);  // 미구현 상태
    EModule Conv2D_Grouped(VDict kwArgs);  // 미구현 상태
    EModule Conv2D_Degormable(VDict kwArgs);  // 미구현 상태
    EModule Deconv(VDict kwArgs = {});
    EModule Max(VDict kwArgs);
    EModule Avg(VDict kwArgs);
    EModule Rnn(VDict kwArgs);
    EModule Lstm(VDict kwArgs);
    EModule Gru(VDict kwArgs);
    EModule Embed(VDict kwArgs);
    EModule Dropout(VDict kwArgs);
    EModule Extract(VDict kwArgs);
    EModule MultiHeadAttention(VDict kwArgs);
    EModule AddBias(VDict kwArgs = {});

    EModule Flatten(VDict kwArgs = {});
    EModule Reshape(VDict kwArgs);
    EModule GlobalAvg(VDict kwArgs = {});
    EModule AdaptiveAvg(VDict kwArgs = {});
    EModule Transpose(VDict kwArgs = {});
    EModule Layernorm(VDict kwArgs = {});
    EModule Batchnorm(VDict kwArgs = {});
    EModule Upsample(VDict kwArgs = {});
    EModule Concat(VDict kwArgs = {});
    EModule Pass(VDict kwArgs = {});
    EModule Noise(VDict kwArgs = {});
    EModule Random(VDict kwArgs = {});
    EModule Round(VDict kwArgs = {});
    EModule CodeConv(VDict kwArgs = {});
    EModule CosineSim(VDict kwArgs = {});
    EModule SelectNTop(VDict kwArgs = {});
    EModule SelectNTopArg(VDict kwArgs = {});

    EModule Activate(VDict kwArgs = {});

    EModule Relu(VDict kwArgs = {});
    EModule Leaky(VDict kwArgs = {});
    EModule Softmax(VDict kwArgs = {});
    EModule Sigmoid(VDict kwArgs = {});
    EModule Tanh(VDict kwArgs = {});
    EModule Gelu(VDict kwArgs = {});
    EModule Mish(VDict kwArgs = {});
    EModule Swish(VDict kwArgs = {});

    EModule Sequential(EModuleList children, VDict kwArgs = {});
    EModule Parallel(EModuleList children, VDict kwArgs = {});
    EModule Add(EModuleList children, VDict kwArgs = {});
    EModule Residual(EModuleList children, VDict kwArgs = {});
    EModule Pruning(EModuleList children, VDict kwArgs = {});
    EModule Stack(EModuleList children, VDict kwArgs = {});
    EModule SqueezeExcitation(EModuleList children, VDict kwArgs = {});

    EModule Formula(string formula, VDict kwArgs = {});

    ELoss MSELoss(VDict kwArgs = {}, string sEstName = "pred", string sAnsName = "y");
    ELoss CrossEntropyLoss(VDict kwArgs = {}, string sLogitName = "pred", string sLabelName = "y");
    ELoss BinaryCrossEntropyLoss(VDict kwArgs = {}, string sLogitName = "pred", string sLabelName = "y");
    ELoss CrossEntropySigmoidLoss(VDict kwArgs = {}, string sLogitName = "pred", string sLabelName = "y");  // will be preciated
    ELoss CrossEntropyPositiveIdxLoss(VDict kwArgs = {}, string sLogitName = "pred", string sLabelName = "y");
    ELoss MultipleLoss(ELossDict losses);
    ELoss CustomLoss(VDict lossTerms, ETensorDict statistics, VDict kwArgs);
    //ELoss CustomLoss(string lossExp);

    EOptimizer createOptimizer(string name, EParameters params, VDict kwArgs);
    EOptimizer SGDOptimizer(EParameters params, VDict kwArgs);
    EOptimizer AdamOptimizer(EParameters params, VDict kwArgs);
    EOptimizer NesterovOptimizer(EParameters params, VDict kwArgs);
    EOptimizer MomentumOptimizer(EParameters params, VDict kwArgs);
    EOptimizer AdaGradOptimizer(EParameters params, VDict kwArgs);
    EOptimizer RMSPropOptimizer(EParameters params, VDict kwArgs);

    EMetric FormulaMetric(string sName, string sFormula, VDict kwArgs);
    EMetric MultipleMetric(EMetricDict metrics, VDict kwArgs);
    EMetric CustomMetric(VDict expTerms, ETensorDict statistics, VDict kwArgs);

protected:
    ELoss m_createLoss(string sBuiltin, VDict kwArgs);
    EMetric m_createMetric(string sBuiltin, VDict kwArgs);

    EFunction m_createFunction(string sBuiltin, string sName, VDict kwArgs);

    EOptimizer m_createOptimizer(string sBuiltin, EParameters params, VDict kwArgs);

    void m_addChidlren(EModule module, EModuleList children);

    EModule m_loadModule(TpStreamIn& fin);

    static int ms_customModuleExecCbFunc(void* pInst, void* pAux, time_t time, VHandle hModule, const VExBuf* pDictBuf, const VExBuf** ppResultBuf);
    static int ms_freeReportBufferCbFunc(void* pInst, void* pAux, const VExBuf* pResultBuf);
};
