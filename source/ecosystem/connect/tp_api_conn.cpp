#include "../connect/tp_api_conn.h"
#include "../connect/tp_http_client_sender.h"
#include "../connect/tp_http_client_receiver.h"
#include "../objects/tp_nn.h"
#include "../objects/tp_module.h"
#include "../objects/tp_tensor.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

mutex ApiConn::ms_serverMutex;
NNRestCbServer* ApiConn::ms_pCbServer = NULL;

/*
void tp_report_error(int errCode, const char* file, int line) {
     print("tp_exception %d at %s:%d", errCode, file, line);
}

void ApiConn::vp_report_error(int errCode, string file, int line, string file_conn, int line_conn) {
    string engineReport = Session_getLastErrorMessage();
    print("engine exception %s ", engineReport.c_str());
    print("   via %s:%d (receive code: %d)", file_conn.c_str(), line_conn, errCode);
    print("   called from %s:%d", file.c_str(), line);
}
*/

ApiConn::ApiConn(ENNCore* nnCore, string server_url, string client_url, VDict kwArgs, string file, int line) {
    m_nnCore = nnCore;
    m_pHttpSender = NULL;
    m_hSession = 0;

    if (server_url == "" || server_url == "127.0.0.1") server_url = "localhost";
    if (server_url != "localhost") {
        if (client_url == "" || client_url == "127.0.0.1") client_url = "http://localhost:8080";

        m_pHttpSender = new NNRestClient(server_url, client_url, this, {});

        ms_serverMutex.lock();
        if (ms_pCbServer == NULL) {
            ms_pCbServer = new NNRestCbServer();
            ms_pCbServer->open(client_url);
            ms_pCbServer->openService();
        }
        ms_pCbServer->setClient(this, m_pHttpSender->getCallbackToken());
        ms_serverMutex.unlock();
    }

    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Session_open(&m_hSession, wrapper.detach()));
    }
}

ApiConn::~ApiConn() {
    try {
        string file = __FILE__;
        int line = __LINE__;

        if (m_pHttpSender == NULL) {
            TP_CALL(V_Session_close(m_hSession));
            eraseSessionHandle(m_hSession);
        }
        else {
            m_pHttpSender->execEngineExec("V_Session_setClose", {});
        }
    }
    catch (...) {
        print("ApiConn::~ApiConn() throws exception: unprocessed");
    }
    
    m_hSession = NULL;
}

/*
void ApiConn::Session_close(string file, int line) {
    TP_CALL(V_Session_close(m_hSession));
    eraseSessionHandle(m_hSession);
}
*/

void ApiConn::login(string username, string password) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("user", "login", { {"username", username}, {"password", password} });
}

void ApiConn::logout() {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("user", "logout", {});
}

void ApiConn::registrate(string username, string password, string email) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("user", "registrate", { {"username", username}, {"password", password}, {"email", email} });
}

VDict ApiConn::getUserInfo(string username) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict response = m_pHttpSender->execTransaction("user", "getinfo", { {"username", username} });
    return response["userinfo"];
}

VList ApiConn::getUserList() {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict response = m_pHttpSender->execTransaction("user", "getlist", {});
    return response["userlist"];
}

void ApiConn::setUserInfo(VDict userInfo, string username) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("user", "setinfo", { {"username", username}, {"userinfo", userInfo} });
}

void ApiConn::closeAccount() {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("user", "close_account", {});
}

void ApiConn::removeUser(string username) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("user", "remove_user", { {"username", username} });
}

VList ApiConn::getRoles() {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict response = m_pHttpSender->execTransaction("role", "get_role_list", {});
    return response["rolelist"];
}

VList ApiConn::getUserRoles(string username) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict response = m_pHttpSender->execTransaction("role", "get_user_roles", { {"username", username} });
    return response["rolelist"];
}

VList ApiConn::getRolePermissions(string rolename) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict response = m_pHttpSender->execTransaction("role", "get_role_perms", { {"rolename", rolename} });
    return response["permlist"];
}

VList ApiConn::getUserPermissions(string username) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict response = m_pHttpSender->execTransaction("role", "get_user_perms", { {"username", username} });
    return response["permlist"];
}

void ApiConn::addRole(string rolename) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("role", "add_role", { {"rolename", rolename} });
}

void ApiConn::remRole(string rolename, bool force) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("role", "rem_role", { {"rolename", rolename}, {"force", force} });
}

void ApiConn::addUserRole(string username, string rolename) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("role", "add_user_role", { {"username", username},  {"rolename", rolename} });
}

void ApiConn::remUserRole(string username, string rolename) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("role", "rem_user_role", { {"username", username},  {"rolename", rolename} });
}

void ApiConn::addRolePermission(string rolename, string permission) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("role", "add_role_perm", { {"rolename", rolename},  {"permission", permission} });
}

void ApiConn::remRolePermission(string rolename, string permission) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    m_pHttpSender->execTransaction("role", "rem_role_perm", { {"rolename", rolename},  {"permission", permission} });
}

void ApiConn::registModel(VHModule hModule, string name, string desc, int type, bool is_public) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict query { {"hmodule", hModule}, {"name", name}, {"desc", desc}, {"type", type}, {"is_public", is_public} };
    m_pHttpSender->execTransaction("model", "regist", query);
}

VList ApiConn::getModelList() {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict response = m_pHttpSender->execTransaction("model", "list", {});
    VList list = response["list"];
    return list;
}

VDict ApiConn::fetchModel(int mid) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict response = m_pHttpSender->execTransaction("model", "fetch", { {"mid", mid} });
    return response;
}

VDict ApiConn::fetchModel(string name) {
    if (m_pHttpSender == NULL) TP_THROW(VERR_INVALID_RESTFUL_SERVER_CORE);
    VDict response = m_pHttpSender->execTransaction("model", "fetch", { {"name", name} });
    return response;
}

void ApiConn::Session_getEngineVersion(string* psEngineVersion, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_getVersion(m_hSession, psEngineVersion));
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Session_getVersion", {});
        *psEngineVersion = response["version"];
    }
}

void ApiConn::Session_seedRandom(int64 random_seed, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_seedRandom(m_hSession, random_seed));
    }
    else {
        m_pHttpSender->execEngineExec("V_Session_seedRandom", { { "random_seed", random_seed } });
    }
}

void ApiConn::Session_setNoGrad(bool no_grad, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_setNoGrad(m_hSession, no_grad));
    }
    else {
        m_pHttpSender->execEngineExec("V_Session_setNoGrad", { { "no_grad", no_grad } });
    }
}

void ApiConn::Session_setNoTracer(bool no_tracer, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_setNoTracer(m_hSession, no_tracer));
    }
    else {
        m_pHttpSender->execEngineExec("V_Session_setNoTracer", { { "no_tracer", no_tracer } });
    }
}

void ApiConn::Session_getCudaDeviceCount(int* pnDeviceCount, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_getCudaDeviceCount(m_hSession, pnDeviceCount));
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Session_getCudaDeviceCount", {});
        *pnDeviceCount = response["device_count"];
    }
}

void ApiConn::Session_getBuiltinNames(VDict* pDict, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pExBuf;
        TP_CALL(V_Session_getBuiltinNames(m_hSession, &pExBuf));
        *pDict = VDictWrapper::unwrap(pExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pExBuf));
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Session_getBuiltinNames", {});
        *pDict = response["builtin_names"];
    }
}

void ApiConn::Session_registCustomModuleExecFunc(VCbCustomModuleExec* pFunc, void* pInst, void* pAux, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_registCustomModuleExecFunc(m_hSession, pFunc, pInst, pAux));
    }
    else {
        m_pHttpSender->execEngineExec("V_Session_registCustomModuleExecFunc", { {"pFunc", (int64)pFunc}, {"pInst", (int64)pInst}, {"pAux", (int64)pAux} });
        //m_pHttpReceiver->registCustomModuleExecFunc(pFunc, pInst, pAux);
    }
}

void ApiConn::Session_registFreeReportBufferFunc(VCbFreeReportBuffer* pFunc, void* pInst, void* pAux, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_registFreeReportBufferFunc(m_hSession, pFunc, pInst, pAux));
    }
    else {
        m_pHttpSender->execEngineExec("V_Session_registFreeReportBufferFunc", { {"pFunc", (int64)pFunc}, {"pInst", (int64)pInst}, {"pAux", (int64)pAux} });
        //m_pHttpReceiver->registFreeReportBufferFunc(pFunc, pInst, pAux);
    }
}

string ApiConn::Session_getFormula(string sBuiltin, string file, int line) {
    if (m_pHttpSender == NULL) {
        string sFormula;
        TP_CALL(V_Session_getFormula(m_hSession, sBuiltin, &sFormula));
        return sFormula;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Session_getFormula", { {"sBuiltin", sBuiltin} });
        string formula = response["formula"];
        return formula;
    }
}

void ApiConn::Session_registMacro(string macroName, VHModule hModule, VDict kwArgs, string file, int line) {
    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Session_registMacro(m_hSession, macroName, hModule, wrapper.detach()));
    }
    else {
        m_pHttpSender->execEngineExec("V_Session_registMacro", { {"macroName", macroName}, {"hModule", hModule}, {"kwArgs", kwArgs} });
    }
}

int ApiConn::Session_addForwardCallbackHandler(TCbForwardCallback* pCbFunc, VDict filters, VDict instInfo, string file, int line) {
    instInfo["#nn_core"] = (int64)m_nnCore;
    instInfo["#cb_func"] = (int64)pCbFunc;

    if (m_pHttpSender == NULL) {
        VDictWrapper fwrapper(filters);
        VDictWrapper iwrapper(instInfo);

        int nId;
        TP_CALL(V_Session_addForwardCallbackHandler(m_hSession, ms_modelCbForwardHandler, ms_modelCbClose, fwrapper.detach(), iwrapper.detach(), &nId));
        return nId;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Session_addForwardCallbackHandler", { {"filters", filters}, {"instInfo", instInfo} });
        int nid = response["nid"];
        return nid;
    }
}

int ApiConn::Session_addBackwardCallbackHandler(TCbBackwardCallback* pCbFunc, VDict filters, VDict instInfo, string file, int line) {
    instInfo["#nn_core"] = (int64)m_nnCore;
    instInfo["#cb_func"] = (int64)pCbFunc;

    if (m_pHttpSender == NULL) {
        VDictWrapper fwrapper(filters);
        VDictWrapper iwrapper(instInfo);

        int nId;
        TP_CALL(V_Session_addBackwardCallbackHandler(m_hSession, ms_modelCbBackwardHandler, ms_modelCbClose, fwrapper.detach(), iwrapper.detach(), &nId));
        return nId;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Session_addBackwardCallbackHandler", { {"filters", filters}, {"instInfo", instInfo} });
        int nid = response["nid"];
        return nid;
    }
}

void ApiConn::Session_removeCallbackHandler(int nId, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_removeCallbackHandler(m_hSession, nId));
    }
    else {
        m_pHttpSender->execEngineExec("V_Session_removeCallbackHandler", { {"nId", nId} });
    }
}

/*
void ApiConn::Session_setUserDefFuncCallback(void* pCbAux, VCbForwardFunction* pForward, VCbBackwardFunction* pBackward, VCbClose* pClose, string file, int line) {
    TP_CALL(V_Session_setFunctCbHandler(m_hSession, pCbAux, pForward, pBackward, pClose));
}
*/

void ApiConn::Session_setUserDefFuncCallback(string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_setFuncCbHandler(m_hSession, m_nnCore, ms_funcCbForwardHandler, ms_funcCbBackwardHandler, ms_funcCbClose));
    }
    else {
        m_pHttpSender->execEngineExec("V_Session_setFuncCbHandler", { {"nn", (int64)m_nnCore}});
    }
}

void ApiConn::Session_freeExchangeBuffer(const VExBuf* pExBuf, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pExBuf));
    }
    else {
        //m_pHttpSender->execEngineExec("V_Session_freeExchangeBuffer", {});
    }
}

VList ApiConn::Session_getLastErrorMessage() {
    VList errorMessages;
    if (m_pHttpSender == NULL) {
        const VExBuf* pErrMessages;
        V_Session_getLastErrorMessageList(m_hSession, &pErrMessages);
        errorMessages = VListWrapper::unwrap(pErrMessages);
        V_Session_freeExchangeBuffer(m_hSession, pErrMessages);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Session_getLastErrorMessage", {});
        errorMessages = response["errorMessages"];
    }
    return errorMessages;
}

int ApiConn::Session_getIdForHandle(VHandle handle, string file, int line) {
    if (m_pHttpSender == NULL) {
        int nId;
        TP_CALL(V_Session_getIdForHandle(m_hSession, handle, &nId));
        return nId;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Session_getIdForHandle", { {"handle", handle} });
        return response["nId"];
    }
}

VDict ApiConn::Session_getLeakInfo(bool sessionOnly, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pLsExBuf;
        TP_CALL(V_Session_getLeakInfo(m_hSession, sessionOnly, &pLsExBuf));
        VDict leakInfo = VDictWrapper::unwrap(pLsExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pLsExBuf));
        return leakInfo;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Debug_dumpLeakInfo", { {"sessionOnly", sessionOnly} });
        return response["leakInfo"];
    }
}

VHLoss ApiConn::Loss_create(string sBuiltin, VDict kwArgs, string file, int line) {
    VHLoss hLoss = 0;

    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Loss_create(m_hSession, &hLoss, sBuiltin, wrapper.detach()));
        registSessionHandle(hLoss);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Loss_create", { {"sBuiltin", sBuiltin}, {"kwArgs", kwArgs} });
        hLoss = response["hLoss"];
    }

    return hLoss;
}

VHMetric ApiConn::Metric_create(string sBuiltin, VDict kwArgs, string file, int line) {
    VHMetric hMetric = 0;

    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Metric_create(m_hSession, &hMetric, sBuiltin, wrapper.detach()));
        registSessionHandle(hMetric);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Metric_create", { {"sBuiltin", sBuiltin}, {"kwArgs", kwArgs} });
        hMetric = response["hMetric"];
    }

    return hMetric;
}

VHFunction ApiConn::Function_create(string sBuiltin, string sName, void* pCbAux, VDict kwArgs, string file, int line) {
    VHFunction hFunction = 0;

    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Function_create(m_hSession, &hFunction, sBuiltin, sName, pCbAux, wrapper.detach()));
        registSessionHandle(hFunction);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Function_create", { {"sBuiltin", sBuiltin}, {"sName", sName}, {"pCbAux", (int64)pCbAux}, {"kwArgs", kwArgs} });
        hFunction = response["hFunction"];
    }

    return hFunction;
}

VHOptimizer ApiConn::Optimizer_create(string sBuiltin, VHParameters hParameters, VDict kwArgs, string file, int line) {
    VHOptimizer hOptimizer = 0;

    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Optimizer_create(m_hSession, &hOptimizer, hParameters, sBuiltin, wrapper.detach()));
        registSessionHandle(hOptimizer);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Optimizer_create", { {"sBuiltin", sBuiltin}, {"hParameters", hParameters}, {"kwArgs", kwArgs} });
        hOptimizer = response["hOptimizer"];
    }

    return hOptimizer;
}

VHModule ApiConn::Module_create(string sBuiltin, string* psName, VDict kwArgs, string file, int line) {
    VHModule hModule = 0;

    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Module_create(m_hSession, &hModule, sBuiltin, psName, wrapper.detach()));
        registSessionHandle(hModule);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_create", { {"sBuiltin", sBuiltin}, {"kwArgs", kwArgs} });
        hModule = response["hModule"];
        *psName = response["sName"];
    }

    return hModule;
}

VHModule ApiConn::Module_createMacro(string sMacroName, string* psName, VDict kwArgs, string file, int line) {
    VHModule hModule = 0;

    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Module_createMacro(m_hSession, &hModule, sMacroName, psName, wrapper.detach()));
        registSessionHandle(hModule);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_createMacro", { {"sMacroName", sMacroName}, {"kwArgs", kwArgs} });
        hModule = response["hModule"];
        *psName = response["sName"];
    }

    return hModule;
}

VHModule ApiConn::Module_createUserDefinedLayer(string name, string formula, VDict paramInfo, VDict kwArgs, string file, int line) {
    VHModule hModule = 0;

    if (m_pHttpSender == NULL) {
        VDictWrapper prapper(paramInfo);
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Module_createUserDefinedLayer(m_hSession, &hModule, name, formula, prapper.detach(), wrapper.detach()));
        registSessionHandle(hModule);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_createUserDefinedLayer", { {"name", name}, {"formula", formula}, {"paramInfo", paramInfo}, {"kwArgs", kwArgs} });
        hModule = response["hModule"];
    }

    return hModule;
}

VHModule ApiConn::Module_load(VDict moduleInfo, string file, int line) {
    VHModule hModule = 0;

    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(moduleInfo);
        TP_CALL(V_Module_load(m_hSession, &hModule, wrapper.detach()));
        registSessionHandle(hModule);
        return hModule;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_load", { {"moduleInfo", moduleInfo} });
        hModule = response["hModule"];
    }

    return hModule;
}

void ApiConn::Module_appendChildModule(VHModule hModule, VHModule hChildModule, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Module_appendChildModule(m_hSession, hModule, hChildModule));
    }
    else {
        m_pHttpSender->execEngineExec("V_Module_appendChildModule", { {"hModule", hModule}, {"hChildModule", hChildModule} });
    }
}

VHTensor ApiConn::Module_evaluate(VHModule hModule, bool train, VHTensor hInput, string file, int line) {
    VHTensor hOutput = 0;
    if (m_pHttpSender == NULL) {
        TP_CALL_TEST(V_Module_evaluate, m_hSession, hModule, train, hInput, &hOutput);
        registSessionHandle(hOutput);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_evaluate", { {"hModule", hModule}, {"train", train}, {"hInput", hInput} });
        hOutput = response["hOutput"];
    }
    return hOutput;
}

VDict ApiConn::Module_evaluateEx(VHModule hModule, bool train, VDict xHandles, string file, int line) {
    VDict yHandles;

    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(xHandles);
        const VExBuf* pYsExBuf;
        TP_CALL(V_Module_evaluateEx(m_hSession, hModule, train, wrapper.detach(), &pYsExBuf));
        yHandles = VDictWrapper::unwrap(pYsExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pYsExBuf));
        for (auto& it : yHandles) {
            registSessionHandle((VHTensor)it.second);
        }
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_evaluateEx", { {"hModule", hModule}, {"train", train}, {"xHandles", xHandles} });
        yHandles = response["yHandles"];
    }

    return yHandles;
}

/*
void ApiConn::Module_loadParameters(VHModule hModule, string filePath, string mode, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Module_loadParameters(m_hSession, hModule, filePath, mode));
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET); // need to upload file contents
        m_pHttpSender->execEngineExec("V_Module_loadParameters", { {"hModule", hModule}, {"filePath", filePath}, {"mode", mode} });
    }
}
*/

void ApiConn::Module_setParamater(VHModule hModule, VDict tHandles, string mode, string file, int line) {
    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(tHandles);
        TP_CALL(V_Module_setParamater(m_hSession, hModule, wrapper.detach(), mode));
    }
    else {
        m_pHttpSender->execEngineExec("V_Module_setParamater", { {"hModule", hModule}, {"tHandles", tHandles}, {"mode", mode} });
    }
}

/*
VDict ApiConn::Module_getSerializeInfo(VHModule hModule, string format, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pInfoBuf;
        TP_CALL(V_Module_getSerializeInfo(m_hSession, hModule, format, &pInfoBuf));
        VDict info = VDictWrapper::unwrap(pInfoBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pInfoBuf));
        return info;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_getSerializeInfo", { {"hModule", hModule}, {"format", format} });
        VDict info = response["info"];
        return info;
    }
}
*/

int ApiConn::Module_addForwardCallbackHandler(VHModule hModule, TCbForwardCallback* pCbFunc, VDict filters, VDict instInfo, string file, int line) {
    instInfo["#nn_core"] = (int64)m_nnCore;
    instInfo["#cb_func"] = (int64)pCbFunc;

    if (m_pHttpSender == NULL) {
        VDictWrapper fwrapper(filters);
        VDictWrapper iwrapper(instInfo);

        int nId;
        TP_CALL(V_Module_addForwardCallbackHandler(m_hSession, hModule, ms_modelCbForwardHandler, ms_modelCbClose, fwrapper.detach(), iwrapper.detach(), &nId));
        return nId;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_addForwardCallbackHandler", { {"hModule", hModule}, {"filters", filters}, {"instInfo", instInfo} });
        int nid = response["nid"];
        return nid;
    }
}

int ApiConn::Module_addBackwardCallbackHandler(VHModule hModule, TCbBackwardCallback* pCbFunc, VDict filters, VDict instInfo, string file, int line) {
    instInfo["#nn_core"] = (int64)m_nnCore;
    instInfo["#cb_func"] = (int64)pCbFunc;

    if (m_pHttpSender == NULL) {
        VDictWrapper fwrapper(filters);
        VDictWrapper iwrapper(instInfo);

        int nId;
        TP_CALL(V_Module_addBackwardCallbackHandler(m_hSession, hModule, ms_modelCbBackwardHandler, ms_modelCbClose, fwrapper.detach(), iwrapper.detach(), &nId));
        return nId;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_addBackwardCallbackHandler", { {"hModule", hModule}, {"filters", filters}, {"instInfo", instInfo} });
        int nid = response["nid"];
        return nid;
    }
}

void ApiConn::Module_removeCallbackHandler(VHModule hModule, int nId, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Module_removeCallbackHandler(m_hSession, hModule, nId));
    }
    else {
        m_pHttpSender->execEngineExec("V_Module_removeCallbackHandler", { {"hModule", hModule}, {"nId", nId} });
    }
}

void ApiConn::Module_uploadDataIndex(VHModule hModule, VList dataIdx, string file, int line) {
    if (m_pHttpSender == NULL) {
        VListWrapper dwrapper(dataIdx);
        TP_CALL(V_Module_uploadDataIndex(m_hSession, hModule, dwrapper.detach()));
    }
    else {
        m_pHttpSender->execEngineExec("V_Module_uploadDataIndex", { {"hModule", hModule}, {"dataIdx", dataIdx} });
    }
}

/*
string ApiConn::Module_getFormula(string sBuiltin, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pInfoBuf;
        TP_CALL(V_Module_getFormula(m_hSession, sBuiltin, &pInfoBuf));
        VDict info = VDictWrapper::unwrap(pInfoBuf);
        return (string)info["formula"];
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_getFormula", { {"sBuiltin", sBuiltin} });
        string formula = response["formula"];
        return formula;
    }
}
*/

VHParameters ApiConn::Module_getParameters(VHModule hModule, string file, int line) {
    VHParameters hParameters = 0;
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Module_getParameters(m_hSession, hModule, &hParameters));
        registSessionHandle(hParameters);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_getParameters", { {"hModule", hModule} });
        hParameters = response["hParameters"];
        
    }
    return hParameters;
}

void ApiConn::Module_copyChildren(VHModule hModule, VHModule hSrcModule, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Module_copyChildren(m_hSession, hModule, hSrcModule));
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_copyChildren", { {"hModule", hModule}, {"hSrcModule", hSrcModule} });
    }
}

VHModule ApiConn::Module_fetchChild(VHModule hModule, string name, bool bChildOnly, string file, int line) {
    if (m_pHttpSender == NULL) {
        VHModule hChildModule;
        TP_CALL(V_Module_fetchChild(m_hSession, hModule, name, bChildOnly, &hChildModule));
        registSessionHandle(hChildModule);
        return hChildModule;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_fetchChild", { {"hModule", hModule}, {"name", name}, {"child_only", bChildOnly} });
        return response["hChildModule"];
    }
}

VList ApiConn::Module_getChildrenModules(VHModule hModule, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pExBuf;
        TP_CALL(V_Module_getChildrenModules(m_hSession, hModule, &pExBuf));
        VList childHandles = VListWrapper::unwrap(pExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pExBuf));
        for (auto& it : childHandles) {
            registSessionHandle((VHTensor)it);
        }
        return childHandles;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_getChildrenModules", { {"hModule", hModule} });
        return response["childHandles"];
    }
}

VHModule ApiConn::Module_expand(VHModule hModule, VShape shape, VDict kwArgs, string file, int line) {
    if (m_pHttpSender == NULL) {
        VHModule hExpandedModule;
        VShapeWrapper swrapper(shape);
        VDictWrapper dwrapper(kwArgs);
        TP_CALL(V_Module_expand(m_hSession, hModule, swrapper.detach(), dwrapper.detach(), &hExpandedModule));
        return hExpandedModule;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_expand", { {"hModule", hModule}, {"shape", shape}, {"kwArgs", kwArgs} });
        return response["hExpandedModule"];
    }
}

/*
VHModule ApiConn::Module_expandMacro(VHModule hModule, VShape shape, VDict kwArgs, string file, int line) {
    if (m_pHttpSender == NULL) {
        VHModule hExpandedModule;
        VShapeWrapper swrapper(shape);
        VDictWrapper dwrapper(kwArgs);
        TP_CALL(V_Module_expandMacro(m_hSession, hModule, swrapper.detach(), dwrapper.detach(), &hExpandedModule));
        registSessionHandle(hExpandedModule);
        return hExpandedModule;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_expandMacro", { {"hModule", hModule}, {"shape", shape}, {"kwArgs", kwArgs} });
        return response["hExpandedModule"];
    }
}
*/

VHModule ApiConn::Module_toDevice(VHModule hModule, string device, string file, int line) {
    if (m_pHttpSender == NULL) {
        VHModule hDeviceModule;
        TP_CALL(V_Module_toDevice(m_hSession, hModule, device, &hDeviceModule));
        return hDeviceModule;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_toDevice", { {"hModule", hModule}, {"device", device} });
        return response["hDeviceModule"];
    }
}

void ApiConn::Module_getModuleInfo(VHModule hModule, VDict dict, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pExBuf = NULL;

        TP_CALL(V_Module_getModuleInfo(m_hSession, hModule, &pExBuf));

        VDict info = VDictWrapper::unwrap(pExBuf);
        for (auto& it : info) {
            dict[it.first] = it.second;
        }

        if (0) {
            // 비어있는 VShape 값을 전달하면 오류가 발생하여 임시조치함 => 처리됨, 추후 문제 없으면 삭제
            if (dict.find("inshape") == dict.end()) dict["inshape"] = VShape();
            if (dict.find("outshape") == dict.end()) dict["outshape"] = VShape();
        }

        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pExBuf));
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Module_getModuleInfo", { {"hModule", hModule} });
        VDict info = response["info"];
        for (auto& it : info) {
            dict[it.first] = it.second;
        }
    }
}

void ApiConn::Module_close(VHModule hModule, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Module_close(m_hSession, hModule));
        eraseSessionHandle(hModule);
    }
    else {
        m_pHttpSender->execEngineExec("V_Module_close", { {"hModule", (int64)hModule} });
    }
}

/*
void ApiConn::Module_setSerializeInfo(VHModule hModule, string format, VDict info, string file, int line) {
    VHSession m_hSession = seekSessionHandle(hModule);
    VDictWrapper wrapper(info);
    TP_CALL(V_Module_setSerializeInfo(m_hSession, hModule, format, wrapper.detach()));
}
*/

VDict ApiConn::Loss_evaluate(VHLoss hLoss, bool download_all, VDict predHandles, VDict yHandles, string file, int line) {
    if (m_pHttpSender == NULL) {
        VDict lossHandles;
        VDictWrapper pwrapper(predHandles);
        VDictWrapper ywrapper(yHandles);
        const VExBuf* pLsExBuf;
        TP_CALL(V_Loss_evaluate(m_hSession, hLoss, download_all, pwrapper.detach(), ywrapper.detach(), &pLsExBuf));
        lossHandles = VDictWrapper::unwrap(pLsExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pLsExBuf));
        for (auto& it : lossHandles) {
            registSessionHandle((VHTensor)it.second);
        }
        return lossHandles;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Loss_evaluate", { {"hLoss", hLoss}, {"download_all", download_all}, {"predHandles", predHandles}, {"yHandles", yHandles} });
        return response["lossHandles"];
    }
}

VDict ApiConn::Loss_eval_accuracy(VHLoss hLoss, bool download_all, VDict predHandles, VDict yHandles, string file, int line) {
    if (m_pHttpSender == NULL) {
        VDict accHandles;
        VDictWrapper pwrapper(predHandles);
        VDictWrapper ywrapper(yHandles);
        const VExBuf* pAccExBuf;
        TP_CALL(V_Loss_eval_accuracy(m_hSession, hLoss, download_all, pwrapper.detach(), ywrapper.detach(), &pAccExBuf));
        accHandles = VDictWrapper::unwrap(pAccExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pAccExBuf));
        for (auto& it : accHandles) {
            registSessionHandle((VHTensor)it.second);
        }
        return accHandles;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Loss_eval_accuracy", { {"hLoss", hLoss}, {"download_all", download_all}, {"predHandles", predHandles}, {"yHandles", yHandles} });
        return response["accHandles"];
    }
}

VDict ApiConn::Metric_evaluate(VHMetric hMetric, VDict pHandles, string file, int line) {
    if (m_pHttpSender == NULL) {
        VDictWrapper pwrapper(pHandles);
        const VExBuf* pLsExBuf;
        TP_CALL(V_Metric_evaluate(m_hSession, hMetric, pwrapper.detach(), &pLsExBuf));
        VDict yHandles = VDictWrapper::unwrap(pLsExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pLsExBuf));
        for (auto& it : yHandles) {
            registSessionHandle((VHTensor)it.second);
        }
        return yHandles;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Metric_evaluate", { {"hMetric", hMetric}, {"pHandles", pHandles} });
        return response["yHandles"];
    }
}

VHTensor ApiConn::Tensor_create(string file, int line) {
    if (m_pHttpSender == NULL) {
        VHTensor hTensor;
        VDictWrapper wrapper(VDict{});
        TP_CALL(V_Tensor_create(m_hSession, &hTensor, "", wrapper.detach()));
        registSessionHandle(hTensor);
        return hTensor;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Tensor_create", {});
        return response["hTensor"];
    }
}

void ApiConn::Tensor_setFeature(VHTensor hTensor, VShape shape, VDataType type, int nDevice, string file, int line) {
    if (m_pHttpSender == NULL) {
        VShapeWrapper wrapper(shape);
        TP_CALL(V_Tensor_setFeature(m_hSession, hTensor, wrapper.detach(), type, nDevice));
    }
    else {
        m_pHttpSender->execEngineExec("V_Tensor_setFeature", { {"hTensor", hTensor}, {"shape", shape} , {"type", (int)type} , {"nDevice", nDevice} });
    }
}

void ApiConn::Tensor_getFeature(VHTensor hTensor, VShape* pshape, VDataType* ptype, int* pnDevice, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pExBuf;
        TP_CALL(V_Tensor_getFeature(m_hSession, hTensor, &pExBuf, ptype, pnDevice));
        *pshape = VShapeWrapper::unwrap(pExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pExBuf));
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Tensor_getFeature", { {"hTensor", hTensor} });
        VShape temp = response["shape"];
        *pshape = temp.copy();
        *ptype = (VDataType)(int)response["type"];
        *pnDevice = response["device"];
    }
}

void ApiConn::Tensor_uploadData(VHTensor hTensor, void* pData, int64 nByteSize, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Tensor_uploadData(m_hSession, hTensor, pData, nByteSize));
    }
    else {
        m_pHttpSender->execEngineExec("V_Tensor_uploadData", { {"hTensor", hTensor} }, pData, nByteSize, true);
    }
}

void ApiConn::Tensor_downloadData(VHTensor hTensor, void* pData, int64 nByteSize, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Tensor_downloadData(m_hSession, hTensor, pData, nByteSize));
    }
    else {
        m_pHttpSender->execEngineExec("V_Tensor_downloadData", { {"hTensor", hTensor}, {"byte_size", nByteSize} }, pData, nByteSize, false);
    }
}

VHTensor ApiConn::Tensor_toDevice(VHTensor hTensor, int nDevice, string file, int line) {
    VHTensor hDevTensor;

    if (m_pHttpSender == NULL) {
        TP_CALL(V_Tensor_toDevice(m_hSession, &hDevTensor, hTensor, nDevice));
        registSessionHandle(hDevTensor);
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Tensor_toDevice", { {"hTensor", hTensor}, {"ndevice", nDevice} });
        hDevTensor = response["hDevTensor"];
    }

    return hDevTensor;
}

void ApiConn::Tensor_backward(VHTensor hTensor, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Tensor_backward(m_hSession, hTensor));
    }
    else {
        m_pHttpSender->execEngineExec("V_Tensor_backward", { {"hTensor", hTensor} });
    }
}

void ApiConn::Tensor_backwardWithGradient(VHTensor hTensor, VHTensor hGrad, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Tensor_backwardWithGradient(m_hSession, hTensor, hGrad));
    }
    else {
        m_pHttpSender->execEngineExec("V_Tensor_backwardWithGradient", { {"hTensor", hTensor}, {"hGrad", hGrad} });
    }
}

void ApiConn::Optimizer_set_option(VHOptimizer hOptimizer, VDict kwArgs, string file, int line) {
    if (m_pHttpSender == NULL) {
        VDictWrapper wrapper(kwArgs);
        TP_CALL(V_Optimizer_set_option(m_hSession, hOptimizer, wrapper.detach()));
    }
    else {
        m_pHttpSender->execEngineExec("V_Optimizer_set_option", { {"hOptimizer", hOptimizer}, {"kwArgs", kwArgs} });
    }
}

void ApiConn::Optimizer_step(VHOptimizer hOptimizer, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Optimizer_step(m_hSession, hOptimizer));
    }
    else {
        m_pHttpSender->execEngineExec("V_Optimizer_step", { {"hOptimizer", hOptimizer} });
    }
}

void ApiConn::Loss_backward(VHLoss hLoss, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Loss_backward(m_hSession, hLoss));
    }
    else {
        m_pHttpSender->execEngineExec("V_Loss_backward", { {"hLoss", hLoss} });
    }
}

void ApiConn::Loss_close(VHLoss hLoss, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Loss_close(m_hSession, hLoss));
        eraseSessionHandle(hLoss);
    }
    else {
        m_pHttpSender->execEngineExec("V_Loss_close", { {"hLoss", (int64)hLoss} });
    }
}

void ApiConn::Metric_close(VHMetric hMetric, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Metric_close(m_hSession, hMetric));
        eraseSessionHandle(hMetric);
    }
    else {
        m_pHttpSender->execEngineExec("V_Metric_close", { {"hMetric", (int64)hMetric} });
    }
}

void ApiConn::Optimizer_close(VHOptimizer hOptimizer, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Optimizer_close(m_hSession, hOptimizer));
        eraseSessionHandle(hOptimizer);
    }
    else {
        m_pHttpSender->execEngineExec("V_Optimizer_close", { {"hOptimizer", (int64)hOptimizer} });
    }
}

void ApiConn::Parameters_close(VHParameters hParameters, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Parameters_close(m_hSession, hParameters));
        eraseSessionHandle(hParameters);
    }
    else {
        m_pHttpSender->execEngineExec("V_Parameters_close", { {"hParameters", (int64)hParameters} });
    }
}

void ApiConn::Parameters_getWeights(VHParameters hParameters, bool bGrad, VList& terms, VDict& tensors, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pListExBuf;
        const VExBuf* pDictExBuf;
        TP_CALL(V_Parameters_getWeights(m_hSession, hParameters, bGrad, &pListExBuf, &pDictExBuf));
        terms = VListWrapper::unwrap(pListExBuf);
        tensors = VDictWrapper::unwrap(pDictExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pListExBuf));
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pDictExBuf));
        for (auto& it : tensors) {
            registSessionHandle((VHTensor)it.second);
        }
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Parameters_getWeights", { {"hParameters", hParameters}, {"grad", bGrad}});
        terms = response["terms"];
        tensors = response["tensors"];
    }
}

/*
VList ApiConn::Parameters_getGradientList(VHParameters hParameters, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pGsExBuf;
        TP_CALL(V_Parameters_getGradientList(m_hSession, hParameters, &pGsExBuf));
        VList infoList = VListWrapper::unwrap(pGsExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pGsExBuf));
        for (auto& it : infoList) {
            registSessionHandle((VHTensor)it["tensor"]);
        }
        return infoList;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("Parameters_getGradientList", { {"hParameters", hParameters} });
        return response["infoList"];
    }
}

VDict ApiConn::Parameters_getWeightDict(VHParameters hParameters, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pWsExBuf;
        TP_CALL(V_Parameters_getWeightDict(m_hSession, hParameters, &pWsExBuf));
        VDict wHandles = VDictWrapper::unwrap(pWsExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pWsExBuf));
        for (auto& it : wHandles) {
            registSessionHandle((VHTensor)it.second);
        }
        return wHandles;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("V_Parameters_getWeights", { {"hParameters", hParameters} });
        return response["wHandles"];
    }
}

VDict ApiConn::Parameters_getGradientDict(VHParameters hParameters, string file, int line) {
    if (m_pHttpSender == NULL) {
        const VExBuf* pGsExBuf;
        TP_CALL(V_Parameters_getGradientDict(m_hSession, hParameters, &pGsExBuf));
        VDict gHandles = VDictWrapper::unwrap(pGsExBuf);
        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pGsExBuf));
        for (auto& it : gHandles) {
            registSessionHandle((VHTensor)it.second);
        }
        return gHandles;
    }
    else {
        VDict response = m_pHttpSender->execEngineExec("Parameters_getGradients", { {"hParameters", hParameters} });
        return response["gHandles"];
    }
}
*/

void ApiConn::Parameters_zeroGrad(VHParameters hParameters, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Parameters_zeroGrad(m_hSession, hParameters));
    }
    else {
        m_pHttpSender->execEngineExec("V_Parameters_zeroGrad", { {"hParameters", hParameters} });
    }
}

void ApiConn::Parameters_initWeights(VHParameters hParameters, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Parameters_initWeights(m_hSession, hParameters));
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET); // need to upload file contents
        m_pHttpSender->execEngineExec("V_Parameters_initWeights", { {"hParameters", hParameters} });
    }
}

void ApiConn::Tensor_close(VHTensor hTensor, string file, int line) {
    if (hTensor == NULL) return;
    if (m_pHttpSender == NULL) {
        if (m_hSession == NULL) return;   // 로컬에서만 이용하는 미등록 텐서의 경우
        TP_CALL(V_Tensor_close(m_hSession, hTensor));
        eraseSessionHandle(hTensor);
    }
    else {
        m_pHttpSender->execEngineExec("V_Tensor_close", { {"hTensor", (int64)hTensor} });
    }
}

void ApiConn::Function_close(VHFunction hFunction, string file, int line) {
    if (m_pHttpSender == NULL) {
        TP_CALL(V_Function_close(m_hSession, hFunction));
        eraseSessionHandle(hFunction);
    }
    else {
        m_pHttpSender->execEngineExec("V_Function_close", { {"hFunction", (int64)hFunction} });
    }
}

ETensor ApiConn::Util_fft(ETensor wave, int64 spec_interval, int64 freq_in_spectrum, int64 fft_width, string file, int line) {
    if (m_pHttpSender == NULL) {
        ETensorDict wTensors({ {"wave", wave} });
        VDict wHandles = TpUtils::TensorDictToDict(wTensors, true);
        VDictWrapper wrapper(wHandles);

        const VExBuf* pRsExBuf;

        TP_CALL(V_Util_fft(m_hSession, wrapper.detach(), spec_interval, freq_in_spectrum, fft_width, &pRsExBuf));

        VDict resultHandles = VDictWrapper::unwrap(pRsExBuf);

        TP_CALL(V_Session_freeExchangeBuffer(m_hSession, pRsExBuf));

        for (auto& it : resultHandles) {
            registSessionHandle((VHTensor)it.second);
        }

        ENN nn(m_nnCore);

        ETensorDict resultTensors = TpUtils::DictToTensorDict(nn, resultHandles);

        ETensor fft = resultTensors["fft"];
        //cats = resultTensors["cats"];

        return fft;
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
        //m_pHttpSender->execEngineExec("V_Function_close", { {"hFunction", (int64)hFunction} });
    }

}

void ApiConn::registSessionHandle(VHandle hHandle) {
    if (m_pHttpSender != NULL) return;

    m_handleMutex.lock();
    if (m_handleRefCntMaps.find(hHandle) != m_handleRefCntMaps.end()) {
        m_handleRefCntMaps[hHandle] = m_handleRefCntMaps[hHandle] + 1;
    }
    else {
        m_handleRefCntMaps[hHandle] = 1;
    }
    m_handleMutex.unlock();
}

void ApiConn::eraseSessionHandle(VHandle hHandle) {
    if (m_pHttpSender != NULL) return;

    m_handleMutex.lock();
    if (m_handleRefCntMaps.find(hHandle) == m_handleRefCntMaps.end()) {
        // do nothing
    }
    else if (m_handleRefCntMaps[hHandle] > 1) {
        m_handleRefCntMaps[hHandle] = m_handleRefCntMaps[hHandle] - 1;
    }
    else {
        m_handleRefCntMaps.erase(hHandle);
    }
    m_handleMutex.unlock();
}

void ApiConn::ms_modelCbForwardHandler(VHSession hSession, const VExBuf* pCbInstBuf, const VExBuf* pCbStatusBuf, const VExBuf* pCbTensorBuf, const VExBuf** ppResultBuf) {
    VDict instInfo = VDictWrapper::unwrap(pCbInstBuf);
    VDict statusInfo = VDictWrapper::unwrap(pCbStatusBuf);

    VDict tensorHandles = VDictWrapper::unwrap(pCbTensorBuf);

    ENN nn((ENNCore*)(int64)instInfo["#nn_core"]);
    TCbForwardCallback* pCbFunc = (TCbForwardCallback*)(int64)instInfo["#cb_func"];

    ETensorDicts tensors = TpUtils::DictToTensorDicts(nn, tensorHandles);

    VDict result = pCbFunc(instInfo, statusInfo, tensors);

    VDictWrapper rwrapper(result);

    *ppResultBuf = rwrapper.detach();
}

void ApiConn::ms_modelCbBackwardHandler(VHSession hSession, const VExBuf* pCbInstBuf, const VExBuf* pCbStatusBuf, const VExBuf* pCbTensorBuf, const VExBuf* pCbGradBuf, const VExBuf** ppResultBuf) {
    VDict instInfo = VDictWrapper::unwrap(pCbInstBuf);
    VDict statusInfo = VDictWrapper::unwrap(pCbStatusBuf);
    VDict tensorHandles = VDictWrapper::unwrap(pCbTensorBuf);
    VDict gradHandles = VDictWrapper::unwrap(pCbGradBuf);

    ENN nn((ENNCore*)(int64)instInfo["#nn_core"]);
    TCbBackwardCallback* pCbFunc = (TCbBackwardCallback*)(int64)instInfo["#cb_func"];

    ETensorDicts tensors = TpUtils::DictToTensorDicts(nn, tensorHandles);
    ETensorDicts grads = TpUtils::DictToTensorDicts(nn, gradHandles);

    VDict result = pCbFunc(instInfo, statusInfo, tensors, grads);

    VDictWrapper rwrapper(result);

    *ppResultBuf = rwrapper.detach();
}

void ApiConn::ms_modelCbClose(VHSession hSession, const VExBuf* pResultBuf) {
    delete pResultBuf;
}

void ApiConn::ms_funcCbForwardHandler(VHSession hSession, void* pHandlerAux, VHFunction hFunction, int nInst, const VExBuf* pTensorListBuf, const VExBuf* pArgDictBuf, const VExBuf** ppResultBuf) {
    //ENN nn((ENNCore*)pHandlerAux);

    VList opndHandles = VListWrapper::unwrap(pTensorListBuf);
    VDict opArgs = VDictWrapper::unwrap(pArgDictBuf);

    //ETensorList operands = TpUtils::ListToTensorList(nn, opndHandles, true);

    //ETensor tensor = nn.funcCbForward(hFunction, nInst, operands, opArgs);

    //ETensorList tensors{ tensor };

    //VList tensorHandles = TpUtils::TensorListToList(tensors, true);

    VList tensorHandles = ApiConn::funcRemoteCbForwardHandler(pHandlerAux, hFunction, nInst, opndHandles, opArgs);

    VListWrapper rwrapper(tensorHandles);

    *ppResultBuf = rwrapper.detach();
}

VList ApiConn::funcRemoteCbForwardHandler(void* pHandlerAux, VHFunction hFunction, int nInst, VList opndHandles, VDict opArgs) {
    ENN nn((ENNCore*)pHandlerAux);

    ETensorList operands = TpUtils::ListToTensorList(nn, opndHandles, true);

    ETensor tensor = nn.funcCbForward(hFunction, nInst, operands, opArgs);

    ETensorList tensors{ tensor };

    VList tensorHandles = TpUtils::TensorListToList(tensors, true);

    return tensorHandles;
}

void ApiConn::ms_funcCbBackwardHandler(VHSession hSession, void* pHandlerAux, VHFunction hFunction, int nInst, const VExBuf* pGradListBuf, const VExBuf* pTensorListBuf, const VExBuf* pArgDictBuf, int nth, const VExBuf** ppResultBuf) {
    VList gradHandles = VListWrapper::unwrap(pGradListBuf);
    VList opndHandles = VListWrapper::unwrap(pTensorListBuf);
    VDict opArgs = VDictWrapper::unwrap(pArgDictBuf);

    VList tensorHandles = funcRemoteCbBackwardHandler(pHandlerAux, hFunction, nInst, gradHandles, opndHandles, opArgs, nth);

    VListWrapper rwrapper(tensorHandles);

    *ppResultBuf = rwrapper.detach();
}

VList ApiConn::funcRemoteCbBackwardHandler(void* pHandlerAux, VHFunction hFunction, int nInst, VList gradHandles, VList opndHandles, VDict opArgs, int nth) {
    ENN nn((ENNCore*)pHandlerAux);

    ETensorList grads = TpUtils::ListToTensorList(nn, gradHandles, true);
    ETensorList operands = TpUtils::ListToTensorList(nn, opndHandles, true);

    ETensor ygrad = grads[0];

    ETensor xgrad = nn.funcCbBackward(hFunction, nInst, ygrad, nth, operands, opArgs);

    ETensorList tensors{ xgrad };

    VList tensorHandles = TpUtils::TensorListToList(tensors, true);

    return tensorHandles;
}

void ApiConn::ms_funcCbClose(VHSession hSession, const VExBuf* pResultBuf) {
    delete pResultBuf;
}

