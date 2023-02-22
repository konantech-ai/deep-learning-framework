#include "../objects/tp_nn.h"
#include "../objects/tp_nn_core.h"
#include "../objects/tp_loss.h"
#include "../objects/tp_module.h"
#include "../objects/tp_tensor.h"
#include "../objects/tp_function.h"
#include "../objects/tp_parameters.h"
#include "../objects/tp_optimizer.h"
#include "../objects/tp_metric.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

//#include "../tconn_python/tp_eco_conn.h"
//#include "../tconn_python/tp_module.h"
//#include "../tconn_python/tp_loss.h"
//#include "../tconn_python/tp_utils.h"

ENN::ENN() { m_core = NULL; }
ENN::ENN(const ENN& src) { m_core = src.m_core->clone(); }
ENN::ENN(ENNCore* core) { m_core = core->clone(); }
ENN::~ENN() { m_core->destroy(); }
ENN& ENN::operator =(const ENN& src) {

    if (&src != this && m_core != src.m_core) {

        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}
bool ENN::isValid() { return m_core != NULL; }
void ENN::close() { if (this) m_core->destroy(); }
ENNCore::ENNCore() : EcoObjCore(VObjType::custom) { m_setup(); }
ENNCore::~ENNCore() { m_delete(); }
ENNCore* ENN::createApiClone() { return m_core->clone(); }

ENN::ENN(string server_url, string client_url) {
    m_core = new ENNCore();

    m_core->m_pApiConn = new ApiConn(m_core, server_url, client_url, {}, __FILE__, __LINE__);

    m_core->m_pApiConn->Session_getEngineVersion(&m_core->m_sEngineVersion, __FILE__, __LINE__);
    m_core->m_pApiConn->Session_getCudaDeviceCount(&m_core->m_nDeviceCount, __FILE__, __LINE__);
    m_core->m_pApiConn->Session_registCustomModuleExecFunc(ms_customModuleExecCbFunc, m_core, NULL, __FILE__, __LINE__);
    m_core->m_pApiConn->Session_registFreeReportBufferFunc(ms_freeReportBufferCbFunc, m_core, NULL, __FILE__, __LINE__);
    m_core->m_pApiConn->Session_getBuiltinNames(&m_core->m_builtinNames, __FILE__, __LINE__);
    m_core->m_pApiConn->Session_setUserDefFuncCallback(__FILE__, __LINE__);
}

int ENN::getEngineObjId(EcoObjCore* pCore) {
    if (pCore == NULL) return -1;
    VHandle handle = pCore->getEngineHandle();
    if (handle == 0) return -1;
    return m_core->m_pApiConn->Session_getIdForHandle(handle, __FILE__, __LINE__);
}

void ENN::login(string username, string password) {
    m_core->m_pApiConn->login(username, password);
}

void ENN::logout() {
    m_core->m_pApiConn->logout();
}

void ENN::registrate(string username, string password, string email) {
    m_core->m_pApiConn->registrate(username, password, email);
}

VList ENN::getUserList() {
    return m_core->m_pApiConn->getUserList();
}

VDict ENN::getUserInfo(string username) {
    return m_core->m_pApiConn->getUserInfo(username);
}

void ENN::setUserInfo(VDict userInfo, string username) {
    m_core->m_pApiConn->setUserInfo(userInfo, username);
}

void ENN::closeAccount() {
    m_core->m_pApiConn->closeAccount();
}

void ENN::removeUser(string username) {
    m_core->m_pApiConn->removeUser(username);
}

VList ENN::getRoles() {
    return m_core->m_pApiConn->getRoles();
}

VList ENN::getUserRoles(string username) {
    return m_core->m_pApiConn->getUserRoles(username);
}

VList ENN::getRolePermissions(string rolename) {
    return m_core->m_pApiConn->getRolePermissions(rolename);
}

VList ENN::getUserPermissions(string username) {
    return m_core->m_pApiConn->getUserPermissions(username);
}

void ENN::addRole(string rolename) {
    m_core->m_pApiConn->addRole(rolename);
}

void ENN::remRole(string rolename, bool force) {
    m_core->m_pApiConn->remRole(rolename, force);
}

void ENN::addUserRole(string username, string rolename) {
    m_core->m_pApiConn->addUserRole(username, rolename);
}

void ENN::remUserRole(string username, string rolename) {
    m_core->m_pApiConn->remUserRole(username, rolename);
}

void ENN::addRolePermission(string rolename, string permission) {
    m_core->m_pApiConn->addRolePermission(rolename, permission);
}

void ENN::remRolePermission(string rolename, string permission) {
    m_core->m_pApiConn->remRolePermission(rolename, permission);
}

void ENN::registModel(EModule model, string name, string desc, bool is_public) {
    int type = (int)model.getType();
    m_core->m_pApiConn->registModel((VHModule) model, name, desc, type, is_public);
}

VList ENN::getModelList() {
    return m_core->m_pApiConn->getModelList();
}

EModule ENN::fetchModel(int mid) {
    VDict response = m_core->m_pApiConn->fetchModel(mid);

    VHModule hModule = response["hModule"];
    string sName = response["name"];
    string sBuiltin = response["builtin"];
    EModuleType moduleType = (EModuleType)(int)response["module_type"];
    VDict kwArgs = response["kwargs"];

    EModule module(nn(), hModule);
    module.setup(sName, sBuiltin, moduleType, kwArgs);

    return module;
}

EModule ENN::fetchModel(string name) {
    VDict response = m_core->m_pApiConn->fetchModel(name);

    VHModule hModule = response["hModule"];
    string sName = response["name"];
    string sBuiltin = response["builtin"];
    EModuleType moduleType = (EModuleType)(int)response["module_type"];
    VDict kwArgs = response["kwargs"];

    EModule module(nn(), hModule);
    module.setup(sName, sBuiltin, moduleType, kwArgs);

    return module;
}

ApiConn* ENN::getApiConn() {
    return m_core->m_pApiConn;
}

void ENN::registTensorHandle(VHTensor hTensor) {
    m_core->m_pApiConn->registSessionHandle(hTensor);
}

string ENN::get_engine_version() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_sEngineVersion;
}

bool ENN::Cuda_isAvailable() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_nDeviceCount > 0;
}

int ENN::Cuda_getDeviceCount() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_nDeviceCount;
}

VDict ENN::get_builtin_names() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_builtinNames;
}

VStrList ENN::get_builtin_names(string domain) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return TpUtils::ListToStrList(m_core->m_builtinNames[domain]);
}

bool ENN::isInBuiltinName(string domain, string sBuiltin) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_builtinNames.find(domain) == m_core->m_builtinNames.end()) {
        TP_THROW(VERR_UNDEFINED);
    }

    VList names = m_core->m_builtinNames[domain];
    for (auto& it : names) {
        if ((string)it == sBuiltin) return true;
    }
    return false;
}

void ENN::srand(int64 random_seed) {
    ::srand((unsigned int)random_seed);
    if (m_core) m_core->m_pApiConn->Session_seedRandom(random_seed, __FILE__, __LINE__);
}

void ENN::set_no_grad() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_pApiConn->Session_setNoGrad(true, __FILE__, __LINE__);
}

void ENN::set_no_tracer() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_pApiConn->Session_setNoTracer(true, __FILE__, __LINE__);
}

void ENN::unset_no_grad() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_pApiConn->Session_setNoGrad(false, __FILE__, __LINE__);
}

void ENN::unset_no_tracer() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_pApiConn->Session_setNoTracer(false, __FILE__, __LINE__);
}

void ENN::saveModule(EModule module, string filename) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

int ENN::addForwardCallbackHandler(TCbForwardCallback* pCbFunc, VDict instInfo, VDict filters) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_pApiConn->Session_addForwardCallbackHandler(pCbFunc, filters, instInfo, __FILE__, __LINE__);
}

int ENN::addBackwardCallbackHandler(TCbBackwardCallback* pCbFunc, VDict instInfo, VDict filters) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_pApiConn->Session_addBackwardCallbackHandler(pCbFunc, filters, instInfo, __FILE__, __LINE__);
}

void ENN::removeCallbackHandler(int nId) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_pApiConn->Session_removeCallbackHandler(nId, __FILE__, __LINE__);
}

VDict ENN::getLeakInfo(bool sessionOnly) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_pApiConn->Session_getLeakInfo(sessionOnly, __FILE__, __LINE__);
}

void ENN::dumpLeakInfo(bool sessionOnly) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VDict leakInfo = m_core->m_pApiConn->Session_getLeakInfo(sessionOnly, __FILE__, __LINE__);
    // 획득한 메모리 릭 정보를 직접 출력하도록 구현한다.
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

EModule ENN::createUserDefinedLayer(string name, string formula, VDict paramInfo, VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VHModule hModule = m_core->m_pApiConn->Module_createUserDefinedLayer(name, formula, paramInfo, kwArgs, __FILE__, __LINE__);

    EModule module(nn(), hModule);
    module.setup(name, "user_defined", EModuleType::layer, kwArgs);

    return module;
}

void ENN::registUserDefFunc(VHFunction hFunction, EFunction* pUserFunc) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    m_core->m_userDefFuncMap[hFunction] = pUserFunc;
}

EModule ENN::Model(string modelName, VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    string sBuiltin = TpUtils::tolower(modelName);
    string sName = TpUtils::seekDict(kwArgs, "name", sBuiltin);

    if (!isInBuiltinName("model", sBuiltin)) TP_THROW(VERR_INVALID_BUILTIN_MODEL);

    VHModule hModule = m_core->m_pApiConn->Module_create(sBuiltin, &sName, kwArgs, __FILE__, __LINE__);

    EModule module(nn(), hModule);
    module.setup(sName, sBuiltin, EModuleType::model, kwArgs);

    return module;
}

EModule ENN::Macro(string macroName, VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    string sName;
    VHModule hModule = m_core->m_pApiConn->Module_createMacro(macroName, &sName, kwArgs, __FILE__, __LINE__);

    EModule module(nn(), hModule);
    module.setup(sName, "macro", EModuleType::macro, kwArgs);

    return module;
}

void ENN::RegistMacro(string macroName, EModule contents, VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_pApiConn->Session_registMacro(macroName, (VHModule)contents, kwArgs, __FILE__, __LINE__);
}

EModule ENN::createModule(string sBuiltin, VDict kwArgs, EModule* pModule) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    EModuleType moduleType;

    sBuiltin = TpUtils::tolower(sBuiltin);

    if (isInBuiltinName("layer", sBuiltin)) {
        moduleType = EModuleType::layer;
    }
    else if (isInBuiltinName("network", sBuiltin)) {
        moduleType = EModuleType::network;
    }
    else {
        TP_THROW(VERR_INVALID_BUILTIN_MODULE);
    }

    string sName;

    VHModule hModule = m_core->m_pApiConn->Module_create(sBuiltin, &sName, kwArgs, __FILE__, __LINE__);

    EModule module(nn(), hModule);
    module.setup(sName, sBuiltin, moduleType, kwArgs);

    return module;
}

VStrList ENN::GetLayerNames() {
    VList names = m_core->m_builtinNames["layer"];
    return TpUtils::ListToStrList(names);
}

string ENN::GetLayerFormula(string layerName) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    string sBuiltin = TpUtils::tolower(layerName);
    string formula;

    if (isInBuiltinName("layer", sBuiltin)) {
        formula = m_core->m_pApiConn->Session_getFormula(sBuiltin, __FILE__, __LINE__);
    }
    else if (isInBuiltinName("network", sBuiltin)) {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return formula;
}

EModule ENN::loadModule(string filePath) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);

    string root = string(getenv("KONANAI_PATH")) + "/work/models/";

    return loadModule(root, filePath);
}

EModule ENN::loadModule(string root, string fileName) {
    string ext = TpUtils::getFileExt(fileName);
    string filePath = root + fileName;

    if (ext == "kon") {
        try {
            TpStreamIn fin(*this, filePath, false);

            if (fin.isOpened()) "open failure";
            if (fin.load_string() != "KonanAI model") throw "bad format";
            if (fin.load_int() != 57578947) throw "bad format";
            if (fin.load_string() != nn().get_engine_version()) throw "bad version";    // 버전 호환성 검사 기능으로 확장 예정 

            EModule model = m_loadModule(fin);

            VShape expandShape = fin.load_shape();

            model.expand(expandShape);

            if (fin.load_string() != "KonanAI param") throw "bad format";
            if (fin.load_int() != 37162886) throw "bad format";
            if (fin.load_string() != nn().get_engine_version()) throw "bad version";    // 버전 호환성 검사 기능으로 확장 예정 

            EParameters params = model.parameters();
            ETensorDict paramsInServer = params.weightDict();

            ETensorDict paramsLoaded = fin.load_tensordict();

            for (auto& it : paramsInServer) {
                printf("param[%s] processing\n", it.first.c_str());
                ETensor src = paramsLoaded[it.first];
                it.second.downloadData();
                if (it.second.hasNoData() && src.hasNoData()) continue;
                it.second.copyData(src.shape(), src.type(), src.void_ptr());
                it.second.upload();
            }

            //TP_THROW(VERR_NOT_IMPLEMENTED_YET);


            return model;
        }
        catch (string sErr) {
            TP_THROW2(VERR_FILE_READ, sErr);
        }
        catch (...) {
            TP_THROW(VERR_FILE_READ);
        }

        /*
        EParameters params = parameters();
        ETensorDict paramsInServer = params.weights();

        for (auto& it : paramsInServer) {
            it.second.downloadData();
            //it.second.dump("[DOWNLOADED] " + it.first);
        }

        fout.save_string("KonanAI model");
        fout.save_int(57578947);
        fout.save_string(nn().get_engine_version());

        m_core->m_saveModel(fout);

        fout.save_string("KonanAI param");
        fout.save_int(37162886);
        fout.save_string(nn().get_engine_version());

        fout.save_tensordict(paramsInServer);
        */
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

EModule ENN::m_loadModule(TpStreamIn& fin) {
    string sBuiltin = fin.load_string();
    VDict kwArgs = fin.load_dict();

    EModule module = createModule(sBuiltin, kwArgs);

    bool nonterm = fin.load_bool();

    if (nonterm) {
        EModuleList children;

        int nChildCount = fin.load_int();

        for (int n = 0; n < nChildCount; n++) {
            children.push_back(m_loadModule(fin));
        }

        m_addChidlren(module, children);
    }
    return module;

}

EModule ENN::loadModule(VDict moduleInfo) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    EModuleType moduleType;

    string sName = moduleInfo["name"];
    string sBuiltin = moduleInfo["builtin"];
    VDict kwArgs = moduleInfo["props"];

    if (isInBuiltinName("layer", sBuiltin)) {
        moduleType = EModuleType::layer;
    }
    else if (isInBuiltinName("network", sBuiltin)) {
        moduleType = EModuleType::network;
    }
    else {
        TP_THROW(VERR_INVALID_BUILTIN_MODULE);
    }

    VHModule hModule = m_core->m_pApiConn->Module_load(moduleInfo, __FILE__, __LINE__);

    EModule module(nn(), hModule);
    module.setup(sName, sBuiltin, moduleType, kwArgs);

    return module;
}

void ENN::registModuleMap(VHModule hModule, EModule* pModule) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_customModuleMap[hModule] = pModule;
}

EModule ENN::Reshape(VDict kwArgs) {
    return createModule("reshape", kwArgs);
}

EModule ENN::Embed(VDict kwArgs) {
    return createModule("embed", kwArgs);
}

EModule ENN::Dropout(VDict kwArgs) {
    return createModule("dropout", kwArgs);
}

EModule ENN::Extract(VDict kwArgs) {
    return createModule("extract", kwArgs);
}

EModule ENN::MultiHeadAttention(VDict kwArgs) {
    return createModule("mh_attention", kwArgs);
}

EModule ENN::AddBias(VDict kwArgs) {
    return createModule("addbias", kwArgs);
}

EModule ENN::Batchnorm(VDict kwArgs) {
    return createModule("batchnorm", kwArgs);
}

EModule ENN::Flatten(VDict kwArgs) {
    return createModule("flatten", kwArgs);
}

EModule ENN::Linear(VDict kwArgs) {
    return createModule("linear", kwArgs);
}

EModule ENN::Dense(VDict kwArgs) {
    return createModule("dense", kwArgs);
}

EModule ENN::Relu(VDict kwArgs) {
    kwArgs["actfunc"] = "relu";
    return createModule("activate", kwArgs);
}

EModule ENN::GlobalAvg(VDict kwArgs) {
    return createModule("globalavg", kwArgs);
}

EModule ENN::AdaptiveAvg(VDict kwArgs) {
    return createModule("adaptiveavg", kwArgs);
}

EModule ENN::Transpose(VDict kwArgs) {
    return createModule("transpose", kwArgs);
}

EModule ENN::Layernorm(VDict kwArgs) {
    return createModule("layernorm", kwArgs);
}

EModule ENN::Upsample(VDict kwArgs) {
    return createModule("upsample", kwArgs);
}

EModule ENN::Concat(VDict kwArgs) {
    return createModule("concat", kwArgs);
}

EModule ENN::Pass(VDict kwArgs) {
    return createModule("pass", kwArgs);
}

EModule ENN::Random(VDict kwArgs) {
    if (kwArgs["method"] == "uniform") {
        return createModule("uniform_random", kwArgs);
    }
    else {
        return createModule("normal_random", kwArgs);
    }
}

EModule ENN::Noise(VDict kwArgs) {
    if (kwArgs["method"] == "uniform") {
        return createModule("uniform_noise", kwArgs);
    }
    else {
        return createModule("normal_noise", kwArgs);
    }
}

EModule ENN::Round(VDict kwArgs) {
    return createModule("round", kwArgs);
}

EModule ENN::CodeConv(VDict kwArgs) {
    return createModule("codeconv", kwArgs);
}

EModule ENN::CosineSim(VDict kwArgs) {
    return createModule("cosinesim", kwArgs);
}

EModule ENN::SelectNTop(VDict kwArgs) {
    return createModule("selectntop", kwArgs);
}

EModule ENN::SelectNTopArg(VDict kwArgs) {
    return createModule("selectntoparg", kwArgs);
}

EModule ENN::Conv(VDict kwArgs) {
    return createModule("conv2d", kwArgs);
}

EModule ENN::Conv1D(VDict kwArgs) {
    return createModule("conv1d", kwArgs);
}

EModule ENN::Conv2D(VDict kwArgs) {
    return createModule("conv2d", kwArgs);
}

EModule ENN::Conv2D_Transposed(VDict kwArgs) {
    return createModule("conv2d_transposed", kwArgs);
}

EModule ENN::Deconv(VDict kwArgs) {
    return createModule("conv2d_transposed", kwArgs);
}

EModule ENN::Conv2D_Separable(VDict kwArgs) {
    TP_THROW(VERR_UNINTENDED);
    return createModule("conv2d_separable", kwArgs);
}

EModule ENN::Conv2D_Depthwise_Separable(VDict kwArgs) {
    TP_THROW(VERR_UNINTENDED);
    return createModule("conv2d_depthwise_separable", kwArgs);
}

EModule ENN::Conv2D_Pointwise(VDict kwArgs) {
    TP_THROW(VERR_UNINTENDED);
    return createModule("conv2d_pointwise", kwArgs);
}

EModule ENN::Conv2D_Grouped(VDict kwArgs) {
    TP_THROW(VERR_UNINTENDED);
    return createModule("conv2d_grouped", kwArgs);
}

EModule ENN::Conv2D_Degormable(VDict kwArgs) {
    TP_THROW(VERR_UNINTENDED);
    return createModule("conv2d_degormable", kwArgs);
}

EModule ENN::Conv2D_Dilated(VDict kwArgs) {
    return createModule("conv2d_dilated", kwArgs);
}

EModule ENN::Max(VDict kwArgs) {
    return createModule("max", kwArgs);
}

EModule ENN::Avg(VDict kwArgs) {
    return createModule("avg", kwArgs);
}

EModule ENN::Rnn(VDict kwArgs) {
    if (kwArgs.find("cell") == kwArgs.end()) kwArgs["cell"] = "rnn";
    return createModule("rnn", kwArgs);
}

EModule ENN::Lstm(VDict kwArgs) {
    if ((string)TpUtils::seekDict(kwArgs, "cell", "lstm") != "lstm") TP_THROW(VERR_CONTENT_DICT_KEY_VALUE);
    return createModule("lstm", kwArgs);
}

EModule ENN::Gru(VDict kwArgs) {
    if ((string)TpUtils::seekDict(kwArgs, "cell", "gru") != "gru") TP_THROW(VERR_CONTENT_DICT_KEY_VALUE);
    return createModule("gru", kwArgs);
}

EModule ENN::Activate(VDict kwArgs) {
    return createModule("activate", kwArgs);
}

EModule ENN::Softmax(VDict kwArgs) {
    return createModule("softmax", kwArgs);
}

EModule ENN::Sigmoid(VDict kwArgs) {
    return createModule("sigmoid", kwArgs);
}

EModule ENN::Tanh(VDict kwArgs) {
    return createModule("tanh", kwArgs);
}

EModule ENN::Mish(VDict kwArgs) {
    return createModule("mish", kwArgs);
}

EModule ENN::Swish(VDict kwArgs) {
    return createModule("swish", kwArgs);
}

EModule ENN::Gelu(VDict kwArgs) {
    return createModule("gelu", kwArgs);
}

EModule ENN::Leaky(VDict kwArgs) {
    return createModule("leaky", kwArgs);
}

void ENN::m_addChidlren(EModule module, EModuleList children) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    for (auto& it : children) {
        EModule child = it;
        if (!child.isValid()) TP_THROW(VERR_INVALID_CORE);
        module.appendChild(child);
    }
}

EModule ENN::Sequential(EModuleList children, VDict kwArgs) {
    EModule module = createModule("sequential", kwArgs);
    m_addChidlren(module, children);
    return module;
}

EModule ENN::Parallel(EModuleList children, VDict kwArgs) {
    EModule module = createModule("parallel", kwArgs);
    m_addChidlren(module, children);
    return module;
}

EModule ENN::Add(EModuleList children, VDict kwArgs) {
    EModule module = createModule("add", kwArgs);
    m_addChidlren(module, children);
    return module;
}

EModule ENN::Residual(EModuleList children, VDict kwArgs) {
    EModule module = createModule("residual", kwArgs);
    m_addChidlren(module, children);
    return module;
}

EModule ENN::Pruning(EModuleList children, VDict kwArgs) {
    EModule module = createModule("pruning", kwArgs);
    m_addChidlren(module, children);
    return module;
}

EModule ENN::Stack(EModuleList children, VDict kwArgs) {
    EModule module = createModule("stack", kwArgs);
    m_addChidlren(module, children);
    return module;
}

EModule ENN::SqueezeExcitation(EModuleList children, VDict kwArgs) {
    EModule module = createModule("squeezeexcitation", kwArgs);
    m_addChidlren(module, children);
    return module;
}

EModule ENN::Formula(string formula, VDict kwArgs) {
    kwArgs["formula"] = formula;
    return createModule("formula", kwArgs);
}

ELoss ENN::m_createLoss(string sBuiltin, VDict kwArgs) {
    sBuiltin = TpUtils::tolower(sBuiltin);

    if (isInBuiltinName("loss", sBuiltin)) {
        VHLoss hLoss = m_core->m_pApiConn->Loss_create(sBuiltin, kwArgs, __FILE__, __LINE__);
        return ELoss(nn(), hLoss);
    }
    else {
        TP_THROW(VERR_INVALID_BUILTIN_LOSS);
    }
}

ELoss ENN::MSELoss(VDict kwArgs, string sEstName, string sAnsName) {
    kwArgs["#estimate"] = sEstName;
    kwArgs["#answer"] = sAnsName;
    return m_createLoss("MSE", kwArgs);
}

ELoss ENN::CrossEntropyLoss(VDict kwArgs, string sLogitName, string sLabelName) {
    kwArgs["#logit"] = sLogitName;
    kwArgs["#label"] = sLabelName;
    return m_createLoss("CrossEntropy", kwArgs);
}

ELoss ENN::BinaryCrossEntropyLoss(VDict kwArgs, string sLogitName, string sLabelName) {
    kwArgs["#logit"] = sLogitName;
    kwArgs["#label"] = sLabelName;
    return m_createLoss("binary_crossentropy", kwArgs);
}

ELoss ENN::CrossEntropySigmoidLoss(VDict kwArgs, string sLogitName, string sLabelName) {
    kwArgs["#logit"] = sLogitName;
    kwArgs["#label"] = sLabelName;
    return m_createLoss("crossentropy_sigmoid", kwArgs);
}

ELoss ENN::CrossEntropyPositiveIdxLoss(VDict kwArgs, string sLogitName, string sLabelName) {
    kwArgs["#logit"] = sLogitName;
    kwArgs["#label"] = sLabelName;
    return m_createLoss("crossentropy_pos_idx", kwArgs);
}

ELoss ENN::MultipleLoss(ELossDict losses) {
    VDict lossHandles = TpUtils::LossDictToDict(losses);;
    VHLoss hLoss = m_core->m_pApiConn->Loss_create("multiple", lossHandles, __FILE__, __LINE__);
    return ELoss(nn(), hLoss, losses);
}

ELoss ENN::CustomLoss(VDict lossTerms, ETensorDict statistics, VDict kwArgs) {
    //return m_createLoss("custom", {{"exp", lossExp}});
    VDict sHandles = TpUtils::TensorDictToDict(statistics, true);
    return m_createLoss("custom", { {"terms", lossTerms}, {"statistics", sHandles}, {"kwArgs", kwArgs} });
}

EMetric ENN::FormulaMetric(string sName, string sFormula, VDict kwArgs) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

EMetric ENN::MultipleMetric(EMetricDict metrics, VDict kwArgs) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

EMetric ENN::CustomMetric(VDict expTerms, ETensorDict statistics, VDict kwArgs) {
    VDict sHandles = TpUtils::TensorDictToDict(statistics, true);
    return m_createMetric("custom", { {"terms", expTerms}, {"statistics", sHandles}, {"kwArgs", kwArgs} });
}

EMetric ENN::m_createMetric(string sBuiltin, VDict kwArgs) {
    sBuiltin = TpUtils::tolower(sBuiltin);

    if (isInBuiltinName("metric", sBuiltin)) {
        VHMetric hMetric = m_core->m_pApiConn->Metric_create(sBuiltin, kwArgs, __FILE__, __LINE__);
        return EMetric(nn(), hMetric);
    }
    else {
        TP_THROW(VERR_INVALID_BUILTIN_LOSS);
    }
}

EOptimizer ENN::createOptimizer(string name, EParameters params, VDict kwArgs) {
    return m_createOptimizer(name, params, kwArgs);
}

EOptimizer ENN::SGDOptimizer(EParameters params, VDict kwArgs) {
    return m_createOptimizer("sgd", params, kwArgs);
}

EOptimizer ENN::AdamOptimizer(EParameters params, VDict kwArgs) {
    return m_createOptimizer("adam", params, kwArgs);
}

EOptimizer ENN::MomentumOptimizer(EParameters params, VDict kwArgs) {
    return m_createOptimizer("momentum", params, kwArgs);
}

EOptimizer ENN::NesterovOptimizer(EParameters params, VDict kwArgs) {
    return m_createOptimizer("nesterov", params, kwArgs);
}

EOptimizer ENN::AdaGradOptimizer(EParameters params, VDict kwArgs) {
    return m_createOptimizer("adagrad", params, kwArgs);
}

EOptimizer ENN::RMSPropOptimizer(EParameters params, VDict kwArgs) {
    return m_createOptimizer("rmsprop", params, kwArgs);
}

EOptimizer ENN::m_createOptimizer(string sBuiltin, EParameters params, VDict kwArgs) {
    sBuiltin = TpUtils::tolower(sBuiltin);

    if (isInBuiltinName("optimizer", sBuiltin)) {
        VHOptimizer hOptimizer = getApiConn()->Optimizer_create(sBuiltin, params, kwArgs, __FILE__, __LINE__);

        EOptimizer optimizer(*this, hOptimizer);

        optimizer.setup(sBuiltin, kwArgs, params);

        params.zero_grad();

        return optimizer;
    }
    else {
        TP_THROW(VERR_INVALID_BUILTIN_OPTIMIZER);
    }
}

EFunction ENN::m_createFunction(string sBuiltin, string sName, VDict kwArgs) {
    return EFunction(*this, sBuiltin, sName, kwArgs);
}

/*
EFunction ENN::ToTensorFunction() {
    return m_createFunction("to_tensor", "to_tensor", {});
}
*/

int ENN::ms_customModuleExecCbFunc(void* pInst, void* pAux, time_t time, VHandle hModule, const VExBuf* pXsBuf, const VExBuf** ppYsBuf) {
    ENNCore* core = (ENNCore*)pInst;
    ENN nn(core);

    try {
        core->m_customModuleExecCbFunc(nn, pAux, time, hModule, pXsBuf, ppYsBuf);
    }
    catch (int nErrCode) {
        return nErrCode;
    }

    return 0;
}

int ENN::ms_freeReportBufferCbFunc(void* pInst, void* pAux, const VExBuf* pResultBuf) {
    ENNCore* core = (ENNCore*)pInst;
    ENN nn(core);

    try {
        core->m_freeReportBufferCbFunc(nn, pAux, pResultBuf);
    }
    catch (int nErrCode) {
        return nErrCode;
    }

    return 0;
}

ETensor ENN::funcCbForward(VHFunction hFunction, int nInst, ETensorList operands, VDict opArgs) {
    EFunction* pFunction = m_core->m_userDefFuncMap[hFunction];

    if (operands.size() == 1) {
        return pFunction->forward(nInst, operands[0], opArgs);
    }
    else {
        return pFunction->forward(nInst, operands, opArgs);
    }
}

ETensor ENN::funcCbBackward(VHFunction hFunction, int nInst, ETensor ygrad, int nth, ETensorList operands, VDict opArgs) {
    EFunction* pFunction = m_core->m_userDefFuncMap[hFunction];

    if (operands.size() == 1) {
        return pFunction->backward(nInst, ygrad, operands[0], opArgs);
    }
    else {
        return pFunction->backward(nInst, ygrad, nth, operands, opArgs);
    }
}

//-----------------------------------------------------------------------------------------------------
// Core part

//map<VHSession, VHandle> ENNCore::ms_sessionToNNMap;

void ENNCore::m_setup() {
}

void ENNCore::m_delete() {
    delete m_pApiConn;
}

void ENNCore::m_customModuleExecCbFunc(ENN nn, void* pAux, time_t time, VHandle hModule, const VExBuf* pXsBuf, const VExBuf** ppYsBuf) {
    if (m_customModuleMap.find(hModule) == m_customModuleMap.end()) TP_THROW(VERR_INVALID_MAP_KEY);

    EModule* pModule = m_customModuleMap[hModule];

    VDict xHandles = VDictWrapper::unwrap(pXsBuf);
    ETensorDict xs = TpUtils::DictToTensorDict(nn, xHandles);

    ETensorDict ys = pModule->forward(xs);
    VDict yHandles = TpUtils::TensorDictToDict(ys, true);

    VDictWrapper wrapper(yHandles);
    *ppYsBuf = wrapper.detach();
}

void ENNCore::m_freeReportBufferCbFunc(ENN nn, void* pAux, const VExBuf* pResultBuf) {
    m_pApiConn->Session_freeExchangeBuffer(pResultBuf, __FILE__, __LINE__);
}

