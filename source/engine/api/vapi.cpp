#define V_EXPORTS

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vmodule.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/vloss.h"
#include "../api_objects/vmetric.h"
#include "../api_objects/voptimizer.h"
#include "../api_objects/vparameters.h"
#include "../api_objects/vfunction.h"
#include "../local_objects/vdevicemanager.h"

VAPI VRetCode V_Session_open(VHSession* phSession, const VExBuf* pDictBuf) {
	try {
		POINTER_CHECK(phSession);

		VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

		VSession session(kwArgs);
		*phSession = session.cloneHandle();
		return VERR_OK;
	}
	catch (...) {
		return VERR_CREATE_SESSION;
	}
}

VAPI VRetCode V_Session_close(VHSession hSession) {
	SESSION_OPEN();

	session.closeObjectInfo();
	session.closeHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_getVersion(VHSession hSession, string * psVersion) {
	SESSION_OPEN();
	*psVersion = session.getVersion();
	SESSION_CLOSE();
}

VAPI VRetCode V_Session_seedRandom(VHSession hSession, int64 rand_seed) {
	SESSION_OPEN();

	session.seedRandom(rand_seed);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_getCudaDeviceCount(VHSession hSession, int* pnDeviceCount) {
	SESSION_OPEN();
	POINTER_CHECK(pnDeviceCount);

	*pnDeviceCount = VDeviceManager::getDeviceCount();

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_getFormula(VHSession hSession, string sBuiltin, string* psFormula) {
	SESSION_OPEN();
	POINTER_CHECK(psFormula);

	VDict result;

	*psFormula = VModule::GetLayerFormula(sBuiltin);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_setNoGrad(VHSession hSession, bool no_grad) {
	SESSION_OPEN();

	session.setNoGrad(no_grad);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_setNoTracer(VHSession hSession, bool no_tracer) {
	SESSION_OPEN();

	session.setNoTracer(no_tracer);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_getDivisions(VHSession hSession, int* pnDivisions) {
	SESSION_OPEN();
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	SESSION_CLOSE();
}

VAPI VRetCode V_Session_getBuiltinNames(VHSession hSession, const VExBuf** ppDictBuf) {
	SESSION_OPEN();

	if (ppDictBuf) {
		VDict builtinNames;

		builtinNames["custom"] = VModule::GetBuiltinCustomNames();
		//builtinNames["model"] = VModule::GetBuiltinModelNames();
		builtinNames["layer"] = VModule::GetBuiltinLayerNames();
		builtinNames["network"] = VModule::GetBuiltinNetworkNames();
		builtinNames["loss"] = VLoss::GetBuiltinNames();
		builtinNames["metric"] = VMetric::GetBuiltinNames();
		builtinNames["optimizer"] = VOptimizer::GetBuiltinNames();
		builtinNames["function"] = VFunction::GetBuiltinNames();

		VDictWrapper wrapper(builtinNames);
		*ppDictBuf = wrapper.detach();
	}

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_registMacro(VHSession hSession, string macroName, VHModule hModule, const VExBuf * pDictBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);

	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	session.registMacro(macroName, module, kwArgs);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_registCustomModuleExecFunc(VHSession hSession, VCbCustomModuleExec * pFunc, void* pInst, void* pAux) {
	SESSION_OPEN();

	session.RegistCustomModuleExecFunc(pFunc, pInst, pAux);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_registFreeReportBufferFunc(VHSession hSession, VCbFreeReportBuffer* pFunc, void* pInst, void* pAux) {
	SESSION_OPEN();

	session.RegistFreeReportBufferFunc(pFunc, pInst, pAux);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_addForwardCallbackHandler(VHSession hSession, VCbForwardModule* pCbFunc, VCbClose* pCbClose, const VExBuf* pFilterBuf, const VExBuf* pCbInstBuf, int* pnId) {
	SESSION_OPEN();

	VDict filter = VDictWrapper::unwrap(pFilterBuf);
	VDict instInfo = VDictWrapper::unwrap(pCbInstBuf);

	int cbItemId = session.addForwardCallbackHandler(pCbFunc, pCbClose, filter, instInfo);

	if (pnId) *pnId = cbItemId;

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_addBackwardCallbackHandler(VHSession hSession, VCbBackwardModule* pCbFunc, VCbClose* pCbClose, const VExBuf* pFilterBuf, const VExBuf* pCbInstBuf, int* pnId) {
	SESSION_OPEN();

	VDict filter = VDictWrapper::unwrap(pFilterBuf);
	VDict instInfo = VDictWrapper::unwrap(pCbInstBuf);

	int cbItemId = session.addBackwardCallbackHandler(pCbFunc, pCbClose, filter, instInfo);

	if (pnId) *pnId = cbItemId;

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_removeCallbackHandler(VHSession hSession, int nId) {
	SESSION_OPEN();

	session.removeCallbackHandler(nId);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_setFuncCbHandler(VHSession hSession, void* pCbAux, VCbForwardFunction* pFuncForward, VCbBackwardFunction* pFuncBackward, VCbClose* pCbClose) {
	SESSION_OPEN();

	session.setFunctionCbHandler(pCbAux, pFuncForward, pFuncBackward, pCbClose);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_freeExchangeBuffer(VHSession hSession, const VExBuf * pBuf) {
	SESSION_OPEN();

	delete pBuf;

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_getLastErrorCode(VHSession hSession, VRetCode * pRetCode) {
	SESSION_OPEN();
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	SESSION_CLOSE();
}

VAPI VRetCode V_Session_getLastErrorMessageList(VHSession hSession, const VExBuf** ppErrMessages) {
	SESSION_OPEN();
	POINTER_CHECK(ppErrMessages);

	VList errMessageList = session.GetLastErrorMessageList();

	VListWrapper wrapper(errMessageList);
	*ppErrMessages = wrapper.detach();

	SESSION_CLOSE();
}


VAPI VRetCode V_Module_create(VHSession hSession, VHModule* phModule, string sBuiltin, string* psName, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	POINTER_CHECK(phModule);

	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	VModule module(session, sBuiltin, "", kwArgs);

	*phModule = module.cloneHandle();
	*psName = module.getName();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_createMacro(VHSession hSession, VHModule* phModule, string sMacroName, string* psName, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	POINTER_CHECK(phModule);

	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	VModule module(session, "macro", sMacroName, kwArgs);

	*phModule = module.cloneHandle();
	*psName = module.getName();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_createUserDefinedLayer(VHSession hSession, VHModule* phModule, string sName, string sFormula, const VExBuf* pParamBuf, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	POINTER_CHECK(phModule);

	VDict paramInfo = VDictWrapper::unwrap(pParamBuf);
	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	VModule module(session, sName, sFormula, paramInfo, kwArgs);
	*phModule = module.cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_load(VHSession hSession, VHModule* phModule, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	POINTER_CHECK(phModule);

	VDict moduleInfo = VDictWrapper::unwrap(pDictBuf);

	VModule module(session, moduleInfo);
	//module.setName(sName);
	*phModule = module.cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_appendChildModule(VHSession hSession, VHModule hModule, VHModule hChildModule) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, parentModule);
	HANDLE_OPEN(VModule, hChildModule, childModule);

	parentModule.appendChild(childModule);

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_evaluate(VHSession hSession, VHModule hModule, bool train, VHTensor hInput, VHTensor * phOutput) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	HANDLE_OPEN(VTensor, hInput, input);
	POINTER_CHECK(phOutput);

	module.macroExpandCheck();

	//input.dump_arr_feat(0, "input");

	VTensor output = module.evaluate(train, input);
	*phOutput = output.cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_evaluateEx(VHSession hSession, VHModule hModule, bool train, const VExBuf* pXsBuf, const VExBuf** ppYsBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	POINTER_CHECK(ppYsBuf);

	VDict xhs = VDictWrapper::unwrap(pXsBuf);
	VTensorDict vars = vutils.toTensorDict(session, xhs);
	if (vars.find("#") == vars.end() && vars.find("x") != vars.end()) {
		vars["#"] = vars["x"];
		vars.erase("x");
	}

	module.macroExpandCheck();

	VTensorDict ys = module.evaluateEx(train, vars);

	VDict yhs = vutils.toDictExternal(ys);

	VDictWrapper wrapper(yhs);
	*ppYsBuf = wrapper.detach();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_getParameters(VHSession hSession, VHModule hModule, VHParameters * phParameters) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	POINTER_CHECK(phParameters);

	module.macroExpandCheck();
	*phParameters = module.getParameters().cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_copyChildren(VHSession hSession, VHModule hModule, VHModule hSrcModule) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	HANDLE_OPEN(VModule, hSrcModule, srcModule);

	module.copyChildren(srcModule);

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_fetchChild(VHSession hSession, VHModule hModule, string name, bool bChildOnly, VHModule* phChildModule) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	POINTER_CHECK(phChildModule);

	module.macroExpandCheck();
	VModule layer = module.fetchChild(name, bChildOnly);

	if (!layer.isValid()) VP_THROW(VERR_UNDEFINED);

	*phChildModule = layer.cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_getChildrenModules(VHSession hSession, VHModule hModule, const VExBuf** ppBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	POINTER_CHECK(ppBuf);

	module.macroExpandCheck();
	VModuleList children = module.getChildrenModules();

	VList chs = vutils.toListExternal(children);

	VListWrapper wrapper(chs);
	*ppBuf = wrapper.detach();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_expand(VHSession hSession, VHModule hModule, const VExBuf* pShapeBuf, const VExBuf* pDictBuf, VHModule* phExpandedModule) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	POINTER_CHECK(phExpandedModule);

	VShape shape = VShapeWrapper::unwrap(pShapeBuf);
	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	*phExpandedModule = module.expand(shape, kwArgs).cloneHandle();

	SESSION_CLOSE();
}

/*
VAPI VRetCode V_Module_expandMacro(VHSession hSession, VHModule hModule, const VExBuf* pShapeBuf, const VExBuf* pDictBuf, VHModule* phExpandedModule) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	POINTER_CHECK(phExpandedModule);

	VShape shape = VShapeWrapper::unwrap(pShapeBuf);
	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	*phExpandedModule = module.expandMacro(shape, kwArgs).cloneHandle();

	SESSION_CLOSE();
}
*/

VAPI VRetCode V_Module_toDevice(VHSession hSession, VHModule hModule, string device, VHModule* phDeviceModule) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);

	//module.macroExpandCheck();

	*phDeviceModule = module.toDevice(device).cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_getModuleInfo(VHSession hSession, VHModule hModule, const VExBuf** ppDictBuf) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VModule, hModule, module);

	module.macroExpandCheck();

	VDict info;

	info["name"] = module.getName();
	info["module_id"] = module.getNth();
	info["module_type"] = (int)module.getModuleType();
	info["nonterm"] = module.getModuleType() != VModuleType::layer;
	info["builtin"] = module.getBuiltIn();
	info["pmsize"] = module.getParamSize();
	info["kwargs"] = module.getKwArgs(); // .cloneCore();
	info["expand_shape"] = module.getExpandShape(); // .cloneCore();

	if (1) {
		VShape inShape = module.getInShape(); // .cloneCore();
		VShape outShape = module.getOutShape(); // .cloneCore();

		if (inShape.size() == 0) inShape = VShape{ 1 };
		if (outShape.size() == 0) outShape = VShape{ 1 }; // .cloneCore();

		info["inshape"] = inShape;
		info["outshape"] = outShape;
	}
	
	if (0) {
		info["inshape"] = module.getInShape(); // .cloneCore();
		info["outshape"] = module.getOutShape(); // .cloneCore();
	}

	VDictWrapper wrapper(info);
	*ppDictBuf = wrapper.detach();
	SESSION_CLOSE();
}

VAPI VRetCode V_Module_loadParameters(VHSession hSession, VHModule hModule, string filePath, string mode) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VModule, hModule, module);

	module.loadParameters(filePath, mode);

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_setParamater(VHSession hSession, VHModule hModule, const VExBuf* pTensorBuf, string mode) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VModule, hModule, module);

	VDict tHandles = VDictWrapper::unwrap(pTensorBuf);

	module.setParamater(tHandles, mode);

	SESSION_CLOSE();
}

/*
VAPI VRetCode V_Module_getSerializeInfo(VHSession hSession, VHModule hModule, string format, const VExBuf** ppDictBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	POINTER_CHECK(ppDictBuf);

	VDict info = module.getSerializeInfo(format);

	VDictWrapper wrapper(info);
	*ppDictBuf = wrapper.detach();

	SESSION_CLOSE();
}
*/

VAPI VRetCode V_Module_addForwardCallbackHandler(VHSession hSession, VHModule hModule, VCbForwardModule* pCbFunc, VCbClose* pCbClose, const VExBuf* pFilterBuf, const VExBuf* pCbInstBuf, int* pnId) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);

	VDict filter = VDictWrapper::unwrap(pFilterBuf);
	VDict instInfo = VDictWrapper::unwrap(pCbInstBuf);

	int cbItemId = module.addForwardCallbackHandler(pCbFunc, pCbClose, filter, instInfo);

	if (pnId) *pnId = cbItemId;

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_addBackwardCallbackHandler(VHSession hSession, VHModule hModule, VCbBackwardModule* pCbFunc, VCbClose* pCbClose, const VExBuf* pFilterBuf, const VExBuf* pCbInstBuf, int* pnId) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);

	VDict filter = VDictWrapper::unwrap(pFilterBuf);
	VDict instInfo = VDictWrapper::unwrap(pCbInstBuf);

	int cbItemId = module.addBackwardCallbackHandler(pCbFunc, pCbClose, filter, instInfo);

	if (pnId) *pnId = cbItemId;

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_removeCallbackHandler(VHSession hSession, VHModule hModule, int nId) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);

	module.removeCallbackHandler(nId);

	SESSION_CLOSE();
}

VAPI VRetCode V_Module_uploadDataIndex(VHSession hSession, VHModule hModule, const VExBuf* pListBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);

	VList dataIdx = VListWrapper::unwrap(pListBuf);

	module.uploadDataIndex(dataIdx);

	SESSION_CLOSE();
}

/*
VAPI VRetCode V_Module_setSerializeInfo(VHSession hSession, VHModule hModule, string format, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VModule, hModule, module);
	POINTER_CHECK(pDictBuf);

	VDict info = VDictWrapper::unwrap(pDictBuf);
	
	module.setSerializeInfo(format, info);

	SESSION_CLOSE();
}
*/

VAPI VRetCode V_Module_close(VHSession hSession, VHModule hModule) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VModule, hModule, module);

	module.closeHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Tensor_create(VHSession hSession, VHTensor * phTensor, string sBuiltin, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	POINTER_CHECK(phTensor);

	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	VTensor tensor(session, sBuiltin, kwArgs);
	*phTensor = (VHTensor)tensor.cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Tensor_close(VHSession hSession, VHTensor hTensor) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VTensor, hTensor, tensor);

	tensor.closeHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Tensor_setFeature(VHSession hSession, VHTensor hTensor, const VExBuf* pShapeBuf, VDataType type, int nDevice) {
	SESSION_OPEN();
	HANDLE_OPEN(VTensor, hTensor, tensor);

	VShape shape = VShapeWrapper::unwrap(pShapeBuf);
	tensor.setFeature(shape, type, nDevice);

	SESSION_CLOSE();
}

VAPI VRetCode V_Tensor_getFeature(VHSession hSession, VHTensor hTensor, const VExBuf** ppShapeBuf, VDataType* ptype, int* pnDevice) {
	SESSION_OPEN();
	HANDLE_OPEN(VTensor, hTensor, tensor);

	VShape shape;
	tensor.getFeature(&shape, ptype, pnDevice);

	if (ppShapeBuf) {
		VShapeWrapper wrapper(shape);
		*ppShapeBuf = wrapper.detach();
	}

	SESSION_CLOSE();
}

VAPI VRetCode V_Tensor_uploadData(VHSession hSession, VHTensor hTensor, void* pData, int64 nByteSize) {
	SESSION_OPEN();
	HANDLE_OPEN(VTensor, hTensor, tensor);

	tensor.uploadData(pData, nByteSize);

	SESSION_CLOSE();
}

VAPI VRetCode V_Tensor_downloadData(VHSession hSession, VHTensor hTensor, void* pData, int64 nByteSize) {
	SESSION_OPEN();
	HANDLE_OPEN(VTensor, hTensor, tensor);

	tensor.downloadData(pData, nByteSize);

	SESSION_CLOSE();
}

VAPI VRetCode V_Tensor_toDevice(VHSession hSession, VHTensor* phTensor, VHTensor hTensor, int nDevice) {
	SESSION_OPEN();
	HANDLE_OPEN(VTensor, hTensor, tensor);

	VTensor clone = tensor.toDevice(nDevice, VExecTracer());
	*phTensor = (VHTensor)clone.cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Tensor_backward(VHSession hSession, VHTensor hTensor) {
	SESSION_OPEN();
	HANDLE_OPEN(VTensor, hTensor, tensor);

	tensor.backward();

	SESSION_CLOSE();
}

VAPI VRetCode V_Tensor_backwardWithGradient(VHSession hSession, VHTensor hTensor, VHTensor hGrad) {
	SESSION_OPEN();
	HANDLE_OPEN(VTensor, hTensor, tensor);

	VTensor grad(session, hGrad);

	tensor.backwardWithGradient(grad);

	SESSION_CLOSE();
}

VAPI VRetCode V_Loss_create(VHSession hSession, VHLoss* phLoss, string sBuiltin, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	POINTER_CHECK(phLoss);

	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	*phLoss = VLoss(session, sBuiltin, kwArgs).cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Loss_close(VHSession hSession, VHLoss hLoss) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VLoss, hLoss, loss);

	loss.closeHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Loss_evaluate(VHSession hSession, VHLoss hLoss, bool download_all, const VExBuf* pPredsBuf, const VExBuf* pYsBuf, const VExBuf** ppLsBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VLoss, hLoss, loss);
	POINTER_CHECK(ppLsBuf);

	VDict phs = VDictWrapper::unwrap(pPredsBuf);
	VDict yhs = VDictWrapper::unwrap(pYsBuf);

	VTensorDict preds = vutils.toTensorDict(session, phs);
	VTensorDict ys = vutils.toTensorDict(session, yhs);

	VTensorDict losses = loss.evaluate(preds, ys, download_all);

	VDict lhs = vutils.toDictExternal(losses);

	VDictWrapper wrapper(lhs);
	*ppLsBuf = wrapper.detach();

	SESSION_CLOSE();
}

VAPI VRetCode V_Loss_eval_accuracy(VHSession hSession, VHLoss hLoss, bool download_all, const VExBuf* pPredsBuf, const VExBuf* pYsBuf, const VExBuf** ppAccBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VLoss, hLoss, loss);
	POINTER_CHECK(ppAccBuf);

	VDict phs = VDictWrapper::unwrap(pPredsBuf);
	VDict yhs = VDictWrapper::unwrap(pYsBuf);

	VTensorDict preds = vutils.toTensorDict(session, phs);
	VTensorDict ys = vutils.toTensorDict(session, yhs);

	VTensorDict accs = loss.eval_accuracy(preds, ys, download_all);

	VDict ahs = vutils.toDictExternal(accs);

	VDictWrapper wrapper(ahs);
	*ppAccBuf = wrapper.detach();

	SESSION_CLOSE();
}

VAPI VRetCode V_Loss_backward(VHSession hSession, VHLoss hLoss) {
	SESSION_OPEN();
	HANDLE_OPEN(VLoss, hLoss, loss);
	
	loss.backward();

	SESSION_CLOSE();
}

VAPI VRetCode V_Metric_create(VHSession hSession, VHMetric* phMetric, string sBuiltin, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	POINTER_CHECK(phMetric);

	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	*phMetric = VMetric(session, sBuiltin, kwArgs).cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Metric_close(VHSession hSession, VHMetric hMetric) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VMetric, hMetric, metric);

	metric.closeHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Metric_evaluate(VHSession hSession, VHMetric hMetric, const VExBuf* ppBuf, const VExBuf** ppLsBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VMetric, hMetric, metric);
	POINTER_CHECK(ppLsBuf);

	VDict phs = VDictWrapper::unwrap(ppBuf);
	VTensorDict preds = vutils.toTensorDict(session, phs);

	VTensorDict metrices = metric.evaluate(preds);

	VDict lhs = vutils.toDictExternal(metrices);

	VDictWrapper wrapper(lhs);
	*ppLsBuf = wrapper.detach();

	SESSION_CLOSE();
}

VAPI VRetCode V_Optimizer_create(VHSession hSession, VHOptimizer * phOptimizer, VHParameters hParameters, string sBuiltin, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VParameters, hParameters, parameters);
	POINTER_CHECK(phOptimizer);

	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	*phOptimizer = VOptimizer(session, parameters, sBuiltin, kwArgs).cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Optimizer_close(VHSession hSession, VHOptimizer hOptimizer) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VOptimizer, hOptimizer, optimizer);

	optimizer.closeHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Optimizer_set_option(VHSession hSession, VHOptimizer hOptimizer, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VOptimizer, hOptimizer, optimizer);

	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	optimizer.set_option(kwArgs);

	SESSION_CLOSE();
}

VAPI VRetCode V_Optimizer_step(VHSession hSession, VHOptimizer hOptimizer) {
	SESSION_OPEN();
	HANDLE_OPEN(VOptimizer, hOptimizer, optimizer);

	optimizer.step();

	SESSION_CLOSE();
}


VAPI VRetCode V_Parameters_close(VHSession hSession, VHParameters hParameters) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VParameters, hParameters, params);

	params.closeHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Parameters_getWeights(VHSession hSession, VHParameters hParameters, bool bGrad, const VExBuf** ppListBuf, const VExBuf** ppDictBuf) {
	SESSION_OPEN();
	HANDLE_OPEN(VParameters, hParameters, parameters);
	POINTER_CHECK(ppListBuf);
	POINTER_CHECK(ppDictBuf);

	VList terms;
	VTensorDict weights;
	
	parameters.getWeights(terms, weights, bGrad);

	VDict ws = vutils.toDictExternal(weights);

	VListWrapper lrapper(terms);
	VDictWrapper drapper(ws);

	*ppListBuf = lrapper.detach();
	*ppDictBuf = drapper.detach();

	SESSION_CLOSE();
}

VAPI VRetCode V_Parameters_zeroGrad(VHSession hSession, VHParameters hParameters) {
	SESSION_OPEN();
	HANDLE_OPEN(VParameters, hParameters, parameters);

	parameters.zero_grad();

	SESSION_CLOSE();
}

VAPI VRetCode V_Parameters_initWeights(VHSession hSession, VHParameters hParameters) {
	SESSION_OPEN();
	HANDLE_OPEN(VParameters, hParameters, parameters);

	parameters.init_weights();

	SESSION_CLOSE();
}

VAPI VRetCode V_Function_create(VHSession hSession, VHFunction* phFunction, string sBuiltin, string sName, void* pCbAux, const VExBuf* pDictBuf) {
	SESSION_OPEN();
	POINTER_CHECK(phFunction);

	VDict kwArgs = VDictWrapper::unwrap(pDictBuf);

	if (sName != "") kwArgs["name"] = sName;
	else sName = (std::string)kwArgs["name"];

	*phFunction = VFunction(session, sBuiltin, sName, pCbAux, kwArgs).cloneHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Function_close(VHSession hSession, VHFunction hFunction) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(VFunction, hFunction, function);

	function.closeHandle();

	SESSION_CLOSE();
}

VAPI VRetCode V_Util_fft(VHSession hSession, const VExBuf* pwBuf, int64 spec_interval, int64 freq_in_spectrum, int64 fft_width, const VExBuf** ppRsBuf) {
	SESSION_OPEN();

	VDict whs = VDictWrapper::unwrap(pwBuf);
	VTensorDict vars = vutils.toTensorDict(session, whs);
	VTensor wave = vars["wave"];

	VTensor fft = session.util_fft(wave, spec_interval, freq_in_spectrum, fft_width);
	
	VTensorDict results{ {"fft", fft} };
	VDict rhs = vutils.toDictExternal(results);
	VDictWrapper rrapper(rhs);
	*ppRsBuf = rrapper.detach();

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_getIdForHandle(VHSession hSession, VHandle handle, int* pnId) {
	SESSION_OPEN();
	POINTER_CHECK(pnId);

	*pnId = session.getIdForHandle(handle);

	SESSION_CLOSE();
}

VAPI VRetCode V_Session_getLeakInfo(VHSession hSession, bool sessionOnly, const VExBuf** ppLsBuf) {
	SESSION_OPEN();

	VDict leakInfo = session.getLeakInfo(sessionOnly);

	VDictWrapper lrapper(leakInfo);
	*ppLsBuf = lrapper.detach();

	SESSION_CLOSE();
}

/*
VAPI VRetCode V_Debug_dumpObjectUsage(string title) {
	extern void dumpObjectUsage(string title);
	dumpObjectUsage(title);
	return 0;
}
*/

VDict V_invokeModuleForwardCallback(VHSession hSession, VCbForwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict) {
	VDictWrapper iwrapper(instInfo);
	VDictWrapper swrapper(statusInfo);
	VDictWrapper twrapper(tensorDict);

	const VExBuf* pResultBuf;

	pCbFunc(hSession, iwrapper.detach(), swrapper.detach(), twrapper.detach(), &pResultBuf);

	VDict result = VDictWrapper::unwrap(pResultBuf);

	pCbClose(hSession, pResultBuf);

	return result;
}

VDict V_invokeModuleBackwardCallback(VHSession hSession, VCbBackwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict, VDict gradDict) {
	VDictWrapper iwrapper(instInfo);
	VDictWrapper swrapper(statusInfo);
	VDictWrapper twrapper(tensorDict);
	VDictWrapper gwrapper(gradDict);

	const VExBuf* pResultBuf;

	pCbFunc(hSession, iwrapper.detach(), swrapper.detach(), twrapper.detach(), gwrapper.detach(), &pResultBuf);

	VDict result = VDictWrapper::unwrap(pResultBuf);

	pCbClose(hSession, pResultBuf);

	return result;
}

VTensor V_invokeFuncForwardCallback(VSession session, void* pHandlerAux, VHFunction hFunction, int nInst, VCbForwardFunction* pFunc, VCbClose* pClose, VTensorList operands, VDict opArgs) {
	VList opndHandles = vutils.toListExternal(operands);

	VListWrapper twrapper(opndHandles);
	VDictWrapper awrapper(opArgs);

	const VExBuf* pResultBuf;

	pFunc(session, pHandlerAux, hFunction, nInst, twrapper.detach(), awrapper.detach(), &pResultBuf);

	VList list = VListWrapper::unwrap(pResultBuf);
	VTensorList tensors = vutils.toTensorList(session, list);
	VTensor y = tensors[0];

	pClose(session, pResultBuf);

	return y;
}

VTensor V_invokeFuncBackwardCallback(VSession session, void* pHandlerAux, VHFunction hFunction, int nInst, VCbBackwardFunction* pFunc, VCbClose* pClose, VTensor ygrad, int nth, VTensorList operands, VDict opArgs) {
	VList gradHandles = vutils.toListExternal(VTensorList{ygrad});
	VList opndHandles = vutils.toListExternal(operands);

	VListWrapper gwrapper(gradHandles);
	VListWrapper twrapper(opndHandles);
	VDictWrapper awrapper(opArgs);

	const VExBuf* pResultBuf;

	pFunc(session, pHandlerAux, hFunction, nInst, gwrapper.detach(), twrapper.detach(), awrapper.detach(), nth, &pResultBuf);

	VList list = VListWrapper::unwrap(pResultBuf);
	VTensorList tensors = vutils.toTensorList(session, list);
	VTensor xgrad = tensors[0];

	pClose(session, pResultBuf);

	return xgrad;
}
