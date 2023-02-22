#pragma once

#include "../utils/tp_common.h"

class EModuleCore;
class ENNCore;
class NNRestClient;
class NNRestCbServer;

struct ModuleInfo {
	string m_sName;
	string m_sBuiltin;

	int64 m_pmSize;

	VShape m_inShape;
	VShape m_outShape;

	VDict m_kwArgs;
};

class ApiConn {
public:
	ApiConn(ENNCore* nnCore, string server_url, string client_url, VDict kwArgs, string file, int line);
	virtual ~ApiConn();

	void vp_report_error(int errCode, string file, int line, string file_conn, int line_conn);

public:
	void login(string username, string password);
	void logout();
	void registrate(string username, string password, string email);
	VList getUserList();
	VDict getUserInfo(string username=""); // username = "" for me, nonzero for others(for administrator only)
	void setUserInfo(VDict userInfo, string username = ""); // username = "" for me, nonzero for others(for administrator only)
	void closeAccount();
	void removeUser(string username);

	VList getRoles();
	VList getUserRoles(string username);
	VList getRolePermissions(string rolename);
	VList getUserPermissions(string username);
	void addRole(string rolename);
	void remRole(string rolename, bool force);
	void addUserRole(string username, string rolename);
	void remUserRole(string username, string rolename);
	void addRolePermission(string rolename, string permission);
	void remRolePermission(string rolename, string permission);

	void registModel(VHModule hModule, string name, string desc, int type, bool is_public);
	VList getModelList();
	VDict fetchModel(int mid);
	VDict fetchModel(string name);

public:
	//void Session_close(VHSession hSession, string file, int line);
	void Session_getEngineVersion(string* psEngineVersion, string file, int line);
	void Session_seedRandom(int64 random_seed, string file, int line);
	void Session_setNoGrad(bool no_grad, string file, int line);
	void Session_setNoTracer(bool no_tracer, string file, int line);
	void Session_getCudaDeviceCount(int* pnDeviceCount, string file, int line);
	void Session_getBuiltinNames(VDict* pDict, string file, int line);
	string Session_getFormula(string sBuiltin, string file, int line);
	void Session_registCustomModuleExecFunc(VCbCustomModuleExec* pFunc, void* pInst, void* pAux, string file, int line);
	void Session_registFreeReportBufferFunc(VCbFreeReportBuffer* pFunc, void* pInst, void* pAux, string file, int line);
	void Session_registMacro(string macroName, VHModule hModule, VDict kwArgs, string file, int line);
	int Session_addForwardCallbackHandler(TCbForwardCallback* pCbFunc, VDict filters, VDict instInfo, string file, int line);
	int Session_addBackwardCallbackHandler(TCbBackwardCallback* pCbFunc, VDict filters, VDict instInfo, string file, int line);
	void Session_removeCallbackHandler(int nId, string file, int line);
	//void Session_setUserDefFuncCallback(VHSession hSession, void* pCbAux, VCbForwardFunction* pForward, VCbBackwardFunction* pBackward, VCbClose* pClose, string file, int line);
	void Session_setUserDefFuncCallback(string file, int line);
	void Session_freeExchangeBuffer(const VExBuf* pExBuf, string file, int line);
	VList Session_getLastErrorMessage();
	int Session_getIdForHandle(VHandle handle, string file, int line);
	VDict Session_getLeakInfo(bool sessionOnly, string file, int line);

	VHModule Module_create(string sBuiltin, string* psName, VDict kwArgs, string file, int line);
	VHModule Module_createMacro(string sMacroName, string* psName, VDict kwArgs, string file, int line);
	VHModule Module_createUserDefinedLayer(string name, string formula, VDict paramInfo, VDict kwArgs, string file, int line);
	VHModule Module_load(VDict moduleInfo, string file, int line);
	//VHModule Module_createClone(VHModule hSrcModule, string file, int line);
	void Module_appendChildModule(VHModule hModule, VHModule hChildModule, string file, int line);
	void Module_close(VHModule hModule, string file, int line);
	VHTensor Module_evaluate(VHModule hModule, bool train, VHTensor x, string file, int line);
	VDict Module_evaluateEx(VHModule hModule, bool train, VDict xHandles, string file, int line);
	VHParameters Module_getParameters(VHModule hModule, string file, int line);
	void Module_copyChildren(VHModule hModule, VHModule hSrcModule, string file, int line);
	VHModule Module_fetchChild(VHModule hModule, string name, bool bChildOnly, string file, int line);
	VList Module_getChildrenModules(VHModule hModule, string file, int line);
	VHModule Module_expand(VHModule hModule, VShape shape, VDict kwArgs, string file, int line);
	//VHModule Module_expandMacro(VHModule hModule, VShape shape, VDict kwArgs, string file, int line);
	VHModule Module_toDevice(VHModule hModule, string device, string file, int line);
	//void Module_getModuleInfo(VHModule hModule, ModuleInfo* pModuleInfo, string file, int line);
	void Module_getModuleInfo(VHModule hModule, VDict dict, string file, int line);
	//void Module_loadParameters(VHModule hModule, string filePath, string mode, string file, int line);
	void Module_setParamater(VHModule hModule, VDict tHandles, string mode, string file, int line);
	//VDict Module_getSerializeInfo(VHModule hModule, string format, string file, int line);
	int Module_addForwardCallbackHandler(VHModule hModule, TCbForwardCallback* pCbFunc, VDict filters, VDict instInfo, string file, int line);
	int Module_addBackwardCallbackHandler(VHModule hModule, TCbBackwardCallback* pCbFunc, VDict filters, VDict instInfo, string file, int line);
	void Module_removeCallbackHandler(VHModule hModule, int nId, string file, int line);
	void Module_uploadDataIndex(VHModule hModule, VList dataIdx, string file, int line);

	//void Module_setSerializeInfo(VHModule hModule, string format, VDict info, string file, int line);

	VHLoss Loss_create(string sBuiltin, VDict kwArgs, string file, int line);
	//VHTensor Loss_evaluate(VHLoss hLoss, VHTensor hPred, VHTensor hY, string file, int line);
	//VHTensor Loss_evaluateEx(VHLoss hLoss, VDict predHandles, VDict yHandles, string file, int line);
	//void Loss_setCustomExpressions(VHLoss hLoss, VDict lossTerms, VDict subTerms, string file, int line);
	VDict Loss_evaluate(VHLoss hLoss, bool download_all, VDict predHandles, VDict yHandles, string file, int line);
	VDict Loss_eval_accuracy(VHLoss hLoss, bool download_all, VDict predHandles, VDict yHandles, string file, int line);
	//void Loss_backward(VHLoss hLoss, VDict lossHandles, string file, int line);
	void Loss_backward(VHLoss hLoss, string file, int line);
	void Loss_close(VHLoss hLoss, string file, int line);

	VHMetric Metric_create(string sBuiltin, VDict kwArgs, string file, int line);
	VDict Metric_evaluate(VHMetric hMetric, VDict pHandles, string file, int line);
	void Metric_close(VHMetric hMetric, string file, int line);

	VHOptimizer Optimizer_create(string sBuiltin, VHParameters hParameters, VDict kwArgs, string file, int line);
	void Optimizer_set_option(VHOptimizer hOptimizer, VDict kwArgs, string file, int line);
	void Optimizer_step(VHOptimizer hOptimizer, string file, int line);
	void Optimizer_close(VHOptimizer hOptimizer, string file, int line);

	void Parameters_getWeights(VHParameters hParameters, bool bGrad, VList& terms, VDict& tensors, string file, int line);
	//void Parameters_getGradients(VHParameters hParameters, string file, int line, VList& terms, VDict& tensors);
	//VList Parameters_getWeightList(VHParameters hParameters, string file, int line);
	//VList Parameters_getGradientList(VHParameters hParameters, string file, int line);
	//VDict Parameters_getWeightDict(VHParameters hParameters, string file, int line);
	//VDict Parameters_getGradientDict(VHParameters hParameters, string file, int line);
	void Parameters_initWeights(VHParameters hParameters, string file, int line);
	void Parameters_zeroGrad(VHParameters hParameters, string file, int line);
	void Parameters_close(VHParameters hParameters, string file, int line);

	VHTensor Tensor_create(string file, int line);
	void Tensor_setFeature(VHTensor hTensor, VShape shape, VDataType dataType, int nDevice, string file, int line);
	void Tensor_getFeature(VHTensor hTensor, VShape* pshape, VDataType* pdataType, int* pnDevice, string file, int line);
	void Tensor_uploadData(VHTensor hTensor, void* pData, int64 nByteSize, string file, int line);
	void Tensor_downloadData(VHTensor hTensor, void* pData, int64 nByteSize, string file, int line);
	VHTensor Tensor_toDevice(VHTensor hTensor, int nDevice, string file, int line);
	void Tensor_backward(VHTensor hTensor, string file, int line);
	void Tensor_backwardWithGradient(VHTensor hTensor, VHTensor hGrad, string file, int line);
	void Tensor_close(VHTensor hTensor, string file, int line);

	VHFunction Function_create(string sBuiltin, string sName, void* pCbAux, VDict kwArgs, string file, int line);
	void Function_close(VHFunction hFunction, string file, int line);

	ETensor Util_fft(ETensor wave, int64 spec_interval, int64 freq_in_spectrum, int64 fft_width, string file, int line);

public:
	//VHSession seekSessionHandle(VHandle hHanle);
	void registSessionHandle(VHandle hHandle);
	void eraseSessionHandle(VHandle hHanle);

protected:
	static void ms_modelCbForwardHandler(VHSession hSession, const VExBuf* pCbInstBuf, const VExBuf* pCbStatusBuf, const VExBuf* pCbTensorBuf, const VExBuf** ppResultBuf);
	static void ms_modelCbBackwardHandler(VHSession hSession, const VExBuf* pCbInstBuf, const VExBuf* pCbStatusBuf, const VExBuf* pCbTensorBuf, const VExBuf* pCbGradBuf, const VExBuf** ppResultBuf);
	static void ms_modelCbClose(VHSession hSession, const VExBuf* pResultBuf);

	static void ms_funcCbForwardHandler(VHSession hSession, void* pHandlerAux, VHFunction hFunction, int nInst, const VExBuf* pTensorListBuf, const VExBuf* pArgDictBuf, const VExBuf** ppResultBuf);
	static void ms_funcCbBackwardHandler(VHSession hSession, void* pHandlerAux, VHFunction hFunction, int nInst, const VExBuf* pGradListBuf, const VExBuf* pTensorListBuf, const VExBuf* pArgDictBuf, int nth, const VExBuf** ppResultBuf);
	static void ms_funcCbClose(VHSession hSession, const VExBuf* pResultBuf);

public:
	static VList funcRemoteCbForwardHandler(void* pHandlerAux, VHFunction hFunction, int nInst, VList opndHandles, VDict opArgs);
	static VList funcRemoteCbBackwardHandler(void* pHandlerAux, VHFunction hFunction, int nInst, VList gradHandles, VList opndHandles, VDict opArgs, int nth);

protected:
	VHSession m_hSession;
	ENNCore* m_nnCore;
	NNRestClient* m_pHttpSender;

	map<VHandle, int> m_handleRefCntMaps;

	mutex m_handleMutex;

	static mutex ms_serverMutex;
	static NNRestCbServer* ms_pCbServer;
};
