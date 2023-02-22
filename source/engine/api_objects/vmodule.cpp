#include <cuda_runtime.h>

#include "../api_objects/vmodule_core.h"
#include "../local_objects/vgraph_core.h"
#include "../local_objects/vexectracerpool.h"
#include "../local_objects/vexectracer_core.h"
#include "../api_objects/vmodule.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/voptimizer.h"
#include "../local_objects/vgraph.h"
#include "../local_objects/vexectracer.h"
#include "../local_objects/vdevicemanager.h"
#include "../api/vconst.h"
#include "../support/vmath.h"
#include "../support/vback_queue.h"
#include "../utils/vutils.h"

int VModuleCore::ms_nCheckCode = 40147291;

//=========== API Object Common Part Start =======================

VModule::VModule() {
	m_core = NULL;
}

VModule::VModule(const VModule& src) {
	m_core = src.m_core->clone();
}

VModule::VModule(VModuleCore* core) {
	m_core = core->clone();
}

VModule::VModule(VSession session, string sBuiltin, VDict kwArgs) {
	m_core = new VModuleCore(session, sBuiltin, kwArgs);
}

VModule::VModule(VSession session, VHModule handle) {
	m_core = NULL;
	VModuleCore* core = (VModuleCore*)handle;
	if (core == NULL) VP_THROW1(VERR_INVALID_CORE, "Module");
	if (core->m_nCheckCode != VModuleCore::ms_nCheckCode) VP_THROW1(VERR_NOT_EQUAL_CORE_CHECKCODE, "Module");
	if (core->m_session != session) VP_THROW1(VERR_NOT_EQUAL_CORE_SESSION, "Module");
	m_core = (VModuleCore*)core->clone_core();
}

VModule::~VModule() { m_core->destroy(); }

VModule& VModule::operator =(const VModule& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VHModule VModule::cloneCore() {
	return (VHModule)m_core->clone();
}

VHModule VModule::cloneHandle() {
	return (VHModule)m_core->clone_handle();
}

VModuleCore* VModule::getClone() {
	return (VModuleCore*)m_core->clone_core();
}

VModuleCore* VModule::getCore() {
	return m_core;
}

bool VModule::isValid() {
	return m_core != NULL;
}
void VModule::closeHandle() {
	if (this) m_core->destroy_handle();
}

VSession VModule::session() {
	return m_core->m_session;
}

int VModule::getRefCnt() {
	return m_core->getRefCnt();
}

int VModule::getNth() {
	return m_core->getNth();
}

void VModule::incRefCount() {
	m_core->incRefCnt();
}

VModuleCore::VModuleCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::Module) {
	m_nCheckCode = ms_nCheckCode;
	m_session = session;
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_allocate = false;
	m_onCreate();
}

VModuleCore::~VModuleCore() {
	m_onDelete();
	m_nCheckCode = 0;
}

//=========== API Object Common Part End =======================

VModule::VModule(VModule& src, bool copyChildren) {
	m_core = new VModuleCore(src.session(), src.m_core->m_sBuiltin, src.m_core->m_propDict);
	m_core->m_setup(src.m_core, copyChildren);
}

VModule::VModule(VSession session, string sBuiltin, string sName, VDict kwArgs) {
	m_core = new VModuleCore(session, sBuiltin, kwArgs);

	m_core->m_sName = vutils.seek_dict(m_core->m_propDict, "name", "");

	if (sBuiltin == "macro") {
		m_core->m_sMacroName = sName;
		int nn = 0;
	}
}

VModule::VModule(VSession session, string sName, string sFormula, VDict paramInfo, VDict kwArgs) {
	m_core = new VModuleCore(session, "user_defined", kwArgs);
	m_core->m_sName = sName;
	m_core->m_setup(sFormula, paramInfo);
}

VModule::VModule(VSession session, VDict moduleInfo) {
	string sBuiltin = moduleInfo["builtin"];
	VDict kwArgs = moduleInfo["props"];

	m_core = new VModuleCore(session, sBuiltin, kwArgs);
	m_core->m_setup(moduleInfo);
}

string VModule::GetLayerFormula(string sBuiltin) {
	return VConsts::getLayerExpression(sBuiltin);
}

void VModule::setName(string sName) {
	m_core->m_sName = sName;
	m_core->m_propDict["name"] = sName;
}

string VModule::getName() {
	return m_core->m_sName;
}

string VModule::getBuiltIn() {
	return m_core->m_sBuiltin;
}

VModuleType VModule::getModuleType() {
	return m_core->m_moduleType;
}

VDict VModule::getKwArgs() {
	return m_core->m_propDict;
}

int64 VModule::getParamSize() {
	return m_core->m_nParamSize;
}

VShape VModule::getInShape() {
	return m_core->m_inShape.copy();
}

VShape VModule::getOutShape() {
	return m_core->m_outShape.copy();
}

VShape VModule::getExpandShape() {
	return m_core->m_shapeExpanded.copy();
}

void VModule::appendChild(VModule child) {
	m_core->m_appendChild(child);
	//string childGivenName = vutils.seek_dict(child.m_core->m_propDict, "name", "");
	//if (childGivenName == "") child.m_core->m_sName = m_core->m_sName + "." + std::to_string(m_core->m_children.size()-1);
}

bool VModule::isUsingCpu() {
	return m_core->m_sDevice == "cpu";
}

void VModule::macroExpandCheck() {
	if (!m_core->m_bMacroExpanded) {
		if (m_core->m_bIncludingMacro) VP_THROW(VERR_UNDEFINED);
		VShape shape;
		VDict shapeDict;
		bool trace = vutils.seek_dict(m_core->m_propDict, "trace", false);
		m_core->m_openGraph(0, shape, shapeDict, trace);
	}
}

VModule VModule::expand(VShape shape, VDict kwArgs) {
	/*
	if (m_core->m_bMacroExpanded) VP_THROW(VERR_UNDEFINED);

	VDict shapeDict = vutils.seek_dict(kwArgs, "xshapes", VDict());

	VModule expandedModule;

	if (m_core->m_bIncludingMacro) {
		expandedModule = m_expandMacroBody(kwArgs);
	}
	else {
		expandedModule = VModule(*this, true);
	}
	*/

	VModule expandedModule = *this;
	VDict shapeDict = vutils.seek_dict(kwArgs, "xshapes", VDict());

	if (!m_core->m_bMacroExpanded) {
		if (m_core->m_bIncludingMacro) {
			expandedModule = m_expandMacroBody(kwArgs);
		}
		else {
			expandedModule = VModule(*this, true);
		}
	}

	expandedModule.setDevice(m_core->m_sDevice);

	bool allocate = vutils.seek_dict(kwArgs, "allocate", true);
	bool trace = vutils.seek_dict(m_core->m_propDict, "trace", false);

	trace = vutils.seek_dict(kwArgs, "trace", trace);

	expandedModule.m_setAllocate(allocate);

	expandedModule.m_core->m_propDict["#xshapes"] = shapeDict;
	expandedModule.m_core->m_openGraph(0, shape, shapeDict, trace);

	return expandedModule;
}

VModule VModule::toDevice(string device) {
	if (m_core->m_sDevice == device) return *this;

	//if (m_core->m_bMacroExpanded) VP_THROW(VERR_UNDEFINED);

	VModule deviceModule(*this, true);

	deviceModule.setDevice(device);

	/*
	VDict shapeDict = vutils.seek_dict(m_core->m_propDict, "#xshapes", VDict());

	bool trace = vutils.seek_dict(m_core->m_propDict, "trace", false);

	deviceModule.m_core->m_openGraph(0, m_core->m_shapeExpanded, shapeDict, trace);
	*/

	return deviceModule;
}

void VModule::setDevice(string device) {
	m_setDevice(device);
}

void VModule::m_setDevice(string device) {
	m_core->m_sDevice = device;

	if (m_core->m_sDevice == "cpu") {
		session().device_man().setUsingCudaFlag(getNth(), 0);
	}
	else if (m_core->m_sDevice == "cuda") {
		session().device_man().setUsingCudaFlag(getNth(), -1);
	}
	else if (m_core->m_sDevice.substr(0, 5) == "cuda:") {
		VStrList devs = vutils.explode(m_core->m_sDevice.substr(5), ",");
		int flag = 0;
		for (auto& it : devs) {
			int n = atoi(it.c_str());
			flag |= 1 << n;
		}
		session().device_man().setUsingCudaFlag(getNth(), flag);
	}
	else {
		VP_THROW1(VERR_INVALID_DEVICE, m_core->m_sDevice);
	}

	for (auto& it : m_core->m_children) {
		it.m_setDevice(device);
	}
}

void VModule::m_setAllocate(bool allocate) {
	m_core->m_allocate = allocate;

	for (auto& it : m_core->m_children) {
		it.m_setAllocate(allocate);
	}
}

VModule VModule::m_expandMacroBody(VDict kwArgs) {
	if (m_core->m_moduleType == VModuleType::macro) {
		VModule macroTemplate = session().getMacro(m_core->m_sMacroName);

		VModule clone(macroTemplate, false);

		clone.setName(m_core->m_sName);

		VDict formalArgs = m_core->m_propDict;

		kwArgs = m_mergeMacroArgs(formalArgs, kwArgs);

		clone.m_core->m_propDict = m_mergeMacroArgs(clone.m_core->m_propDict, kwArgs);

		for (auto& it : macroTemplate.m_core->m_children) {
			clone.m_core->m_children.push_back(it.m_expandMacroBody(kwArgs));
		}
		return clone;
	}
	else if (m_core->m_bIncludingMacro) {
		VModule clone(*this, false);

		VDict formalArgs = m_core->m_propDict;
		//kwArgs = m_mergeMacroArgs(formalArgs, kwArgs, clone);
		VDict actualArgs = m_mergeMacroArgs(formalArgs, kwArgs);

		clone.m_core->m_propDict = actualArgs;

		for (auto& it : m_core->m_children) {
			clone.m_core->m_children.push_back(it.m_expandMacroBody(kwArgs));
		}
		return clone;
	}
	else {
		//VModule clone(*this, true);
		VModule clone(*this, false);

		VDict formalArgs = m_core->m_propDict;
		//kwArgs = m_mergeMacroArgs(formalArgs, kwArgs);

		clone.m_core->m_propDict = m_mergeMacroArgs(clone.m_core->m_propDict, kwArgs);

		for (auto& it : m_core->m_children) {
			clone.m_core->m_children.push_back(it.m_expandMacroBody(kwArgs));
		}

		return clone;
	}
}

VModule VModule::m_cloneModuleBody() {
	VModule clone(*this, false);

	for (auto& it : m_core->m_children) {
		clone.m_core->m_children.push_back(it.m_cloneModuleBody());
	}

	return clone;
}

VDict VModule::m_mergeMacroArgs(VDict formalArgs, VDict& actualArgs) {
	VDict mergedArgs;
	VList usedName;

	if (vutils.seek_dict(formalArgs, "#inherit_all", false) || vutils.seek_dict(formalArgs, "inherit_all", false)) {
		for (auto& it : actualArgs) {
			mergedArgs[it.first] = it.second;
		}

		for (auto& it : formalArgs) {
			mergedArgs[it.first] = it.second;
		}

		return mergedArgs;
	}

	for (auto& it : formalArgs) {
		if (!it.second.is_string() || ((string)it.second)[0] != '#') {
			mergedArgs[it.first] = it.second;
			usedName.push_back(it.first);
		}
		else {
			string key = ((string)it.second).substr(1);
			if (key == "") key = it.first;

			if (actualArgs.find(key) != actualArgs.end()) {
				mergedArgs[it.first] = actualArgs[key];
				usedName.push_back(key);
				//actualArgs.erase(key);
				if (it.first == "set" || it.first == "get") actualArgs.erase(key);
			}
			else {
				// {"chn", "#chn1*2"}, {"chn", "#chn*2"} 등을 처리해보자.
				std::size_t found_mul = key.find('*');
				std::size_t found_div = key.find('/');

				if (found_mul != std::string::npos) {
					string subkey = key.substr(0, found_mul);
					string multiplier = key.substr(found_mul + 1);
					if (actualArgs.find(subkey) != actualArgs.end()) {
						int64 val = actualArgs[subkey];
						int64 coef = std::stoi(multiplier);
						
						mergedArgs[it.first] = val * coef;
						usedName.push_back(subkey);
					}
				}
				else if (found_div != std::string::npos) {
					string subkey = key.substr(0, found_div);
					string divisior = key.substr(found_div + 1);
					if (actualArgs.find(subkey) != actualArgs.end()) {
						int64 val = actualArgs[subkey];
						int64 coef = std::stoi(divisior);

						mergedArgs[it.first] = val / coef;
						usedName.push_back(subkey);
					}
				}

				int nnn = 0; // 값이 제공되지 않은 형식인자: 무시하고 지나가는 게 맞을 듯
			}
		}
	}

	// formalArgs에서 언급되지 않은 actualArgs 항목들을 따로 추가할 필요가 있을지도 모름
	// 그러나 아래와 같은 일괄 전달은 불필요한 요소에 잘못된 정보를 전달하여 실행을 엉망으로 만듦
	/*
	for (auto& it : actualArgs) {
		if (mergedArgs.find(it.first) == mergedArgs.end()) {
			mergedArgs[it.first] = it.second;
		}
	}
	*/
	
	// 단말 레이어는 불필요한 정보를 받으면 안되지만 중간 전달하는 노드들은 전달해보자.
	if (m_core->m_moduleType != VModuleType::layer) {
		for (auto& it : actualArgs) {
			if (mergedArgs.find(it.first) == mergedArgs.end()) {
				if (it.second.is_string() && ((string)it.second)[0] == '#') continue;
				if (usedName.find(it.first) != usedName.end()) continue;
				mergedArgs[it.first] = it.second;
			}
		}
	}

	return mergedArgs;
}

VList VModule::params() {
	return m_core->m_params;
}

VParameters VModule::getParameters() {
	return m_core->m_getParameters();
}

void VModule::setParamater(VDict tHandles, string mode) {
	VTensorDict tensors = vutils.toTensorDict(session(), tHandles);

	m_core->m_setParamater(tensors, mode);
}

/*
void VModuleCore::m_setParamater(VTensorDict tensors, string mode) {
	if (mode == "kai") m_setParamaterKai(tensors);
	else if (mode == "torch") m_setParamaterTorch(tensors);
	else VP_THROW(VERR_UNDEFINED);
}

void VModuleCore::m_setParamaterKai(VTensorDict tensors) {

	for (auto& it : m_children) {
		it.m_core->m_setParamaterKai(tensors);
	}
}
*/

void VModuleCore::m_setParamater(VTensorDict tensors, string mode) {
	string nameDot = m_sName + ".";
	//if (m_sBuiltin == "lstm") nameDot = "rnn.";

	bool matched = false;

	for (auto& it : tensors) {
		if (it.first.rfind(nameDot, 0) == 0) {
			if (m_sBuiltin == "rnn" || m_sBuiltin == "lstm" || m_sBuiltin == "gru") {
				/*
				if (m_sBuiltin == "lstm") ngate = 4;
				else if (m_sBuiltin == "gru") ngate = 3;
				*/

				VDict pmset = m_params[0];

				int nLayers = pmset["num_layers"];
				bool bidir = pmset["bidirectional"];

				for (int n = 0; n < nLayers; n++) {
					string wi_name = "L" + std::to_string(n) + "F_i";
					string wr_name = "L" + std::to_string(n) + "F_r";

					string src_wi_name = nameDot + "weight_ih_l" + std::to_string(n) + ".data.npy";
					string src_wr_name = nameDot + "weight_hh_l" + std::to_string(n) + ".data.npy";

					m_setMatchedParam(pmset, tensors, wi_name, src_wi_name, mode);
					m_setMatchedParam(pmset, tensors, wr_name, src_wr_name, mode);

					if (bidir) {
						string wi_name = "L" + std::to_string(n) + "R_i";
						string wr_name = "L" + std::to_string(n) + "R_r";

						string src_wi_name = nameDot + "weight_ih_l" + std::to_string(n) + "_reverse.data.npy";
						string src_wr_name = nameDot + "weight_hh_l" + std::to_string(n) + "_reverse.data.npy";

						m_setMatchedParam(pmset, tensors, wi_name, src_wi_name, mode);
						m_setMatchedParam(pmset, tensors, wr_name, src_wr_name, mode);
					}
				}
			}
			else if (m_sBuiltin == "linear") {
				VDict pmset = m_params[0];
				m_setMatchedParam(pmset, tensors, "", nameDot + "weight.data.npy", mode);
			}
			else if (m_sBuiltin == "conv2d") {
				VDict pmset = m_params[0];
				m_setMatchedParam(pmset, tensors, "", nameDot + "weight.data.npy", mode);
			}
			else if (m_sBuiltin == "sequential") {
				int nnn = 0;
			}
			else {
				//printf("TTIMPLE: m_sBuiltin = %s\n", m_sBuiltin.c_str());
				int nnn = 0;
			}

			matched = true;

			break;
		}
	}

	if (!matched) {
		if (m_sBuiltin == "sequential" || m_sBuiltin == "activate" || m_sBuiltin == "dropout" ||
			m_sBuiltin == "max" || m_sBuiltin == "adaptiveavg" || m_sBuiltin == "transpose") {
		}
		else {
			//printf("unmatched: m_sBuiltin = %s, nameDot = %s\n", m_sBuiltin.c_str(), nameDot.c_str());
		}
		int nnn = 0;
	}

	for (auto& it : m_children) {
		it.m_core->m_setParamater(tensors, mode);
	}
}

void VModuleCore::m_setMatchedParam(VDict pmset, VTensorDict tensors, string dstName, string srcName, string mode) {
	VDict pm_w = pmset[dstName+"w"];
	VTensor w((VTensorCore*)(VObjCore*)pm_w["pm"]);

	if (mode == "kai" && dstName == "") srcName = m_sName + ".w";
	if (tensors.find(srcName) == tensors.end()) VP_THROW(VERR_UNDEFINED);
	w.copyParam(tensors[srcName], mode);

	if ((bool)pmset["use_bias"]) {
		VDict pm_b = pmset[dstName + "b"];
		VTensor b((VTensorCore*)(VObjCore*)pm_b["pm"]);
		if (mode == "kai" && dstName == "") srcName = m_sName + ".b";
		else if (mode == "torch") srcName.replace(srcName.find("weight"), 6, "bias");
		if (tensors.find(srcName) == tensors.end()) VP_THROW(VERR_UNDEFINED);
		b.copyParam(tensors[srcName], mode);
	}
}

void VModule::loadParameters(string filePath, string mode) {
	FILE* fid = vutils.fopen(filePath, "rb");
	if (fid == NULL) VP_THROW(VERR_FILE_OPEN);

	int major, minor, revision;
	uint64_t iseen = 0;

	fread(&major, sizeof(int), 1, fid);
	fread(&minor, sizeof(int), 1, fid);
	fread(&revision, sizeof(int), 1, fid);

	fread(&iseen, sizeof(uint64_t), 1, fid);

	// 아래 조합은 darknet의 yolo4.weights 파라미터의 경우에 해당하며 일반화 방법은 모르는 상태임
	if (major != 0 || minor != 2 || revision != 5) VP_THROW(VERR_UNDEFINED);

	m_loadParameters(fid, m_core->m_params);
#ifdef FOR_LINUX
	#define _ftelli64 ftello
	#define _fseeki64 fseeko
#endif
	int64 pos1 = _ftelli64(fid);
	_fseeki64(fid, 0, SEEK_END);
	int64 pos2 = _ftelli64(fid);
	
	if (pos1 != pos2) VP_THROW(VERR_UNDEFINED);

	fclose(fid);
}

void VModule::m_loadParameters(FILE* fid, VList params) {
	for (auto& it : params) {
		if (it.is_list()) m_loadParameters(fid, (VList)it);
		else if (it.is_dict()) m_loadParameters(fid, (VDict)it);
		else {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
	}
}

void VModule::m_loadParameters(FILE* fid, VDict params) {
	string type = params["type"];

	if (type == "conv2d") {
		VParameters::load_param(params["b"], "bias", fid);

		VParameters::load_param(params["w"], "kernel", fid);

		VDict pmset = params["w"];
		VTensor pm((VTensorCore*)(VObjCore*)pmset["pm"]);

		string sdesc = pm.shape().desc();

		static int64 nth = 0;
		int64 pos = _ftelli64(fid);
	}
	else if (type == "args") {
		// 파라미터 적재 불필요하므로 통과
	}
	else {
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}
}

void VModule::setMacroArgs(VDict kwArgs) {
	m_core->m_macroArgs = kwArgs;
}

VDict VModule::getSerializeInfo(string format) {
	if (format == "onnx") VP_THROW(VERR_NOT_IMPLEMENTED_YET);

	bool bIncludeParam = (format == "kmp");

	return m_core->m_getSerializeInfo(bIncludeParam);
}

int VModule::addForwardCallbackHandler(VCbForwardModule* pCbFunc, VCbClose* pCbClose, VDict filters, VDict instInfo) {
	VCbItem cbItem(session(), pCbFunc, pCbClose, filters, instInfo);

	m_core->m_cbForwardItemMap[cbItem.getNth()] = cbItem;
	m_core->m_bNeedForwardCallbeck = true;

	return cbItem.getNth();
}

int VModule::addBackwardCallbackHandler(VCbBackwardModule* pCbFunc, VCbClose* pCbClose, VDict filters, VDict instInfo) {
	VCbItem cbItem(session(), pCbFunc, pCbClose, filters, instInfo);

	m_core->m_cbBackwardItemMap[cbItem.getNth()] = cbItem;
	m_core->m_bNeedBackwardCallbeck = true;

	return cbItem.getNth();
}

void VModule::removeCallbackHandler(int nId) {
	if (m_core->m_cbForwardItemMap.find(nId) != m_core->m_cbForwardItemMap.end()) {
		m_core->m_cbForwardItemMap.erase(nId);
		m_core->m_bNeedForwardCallbeck = (m_core->m_cbForwardItemMap.size() > 0);
	}
	else if (m_core->m_cbBackwardItemMap.find(nId) != m_core->m_cbBackwardItemMap.end()) {
		m_core->m_cbBackwardItemMap.erase(nId);
		m_core->m_bNeedBackwardCallbeck = (m_core->m_cbBackwardItemMap.size() > 0);
	}
	else {
		VP_THROW(VERR_INVALID_MAP_KEY);
	}
}

void VModule::uploadDataIndex(VList dataIdx) {
	if (m_core->m_dataIdx.size() != dataIdx.size()) {
		m_core->m_dataIdx = dataIdx;
	}
	else {
		for (int64 n = 0; n < dataIdx.size(); n++) {
			m_core->m_dataIdx[n] = dataIdx[n];
		}
	}
}

void VModule::copyChildren(VModule srcModule) {
	for (auto& it : srcModule.m_core->m_children) {
		VModule clone(it, true);
		m_core->m_children.push_back(clone);
	}
}

VModule VModule::fetchChild(string name, bool bChildOnly) {
	for (auto& it : m_core->m_children) {
		if (it.getName() == name) {
			return it;
		}
		if (!bChildOnly && it.m_core->m_children.size() > 0) {
			VModule layer = it.fetchChild(name, false);
			if (layer.isValid()) return layer;
		}
	}

	return VModule();
}

VModuleList VModule::getChildrenModules() {
	return m_core->m_children;
}

VTensor VModule::evaluate(bool train, VTensor x) {
	if (!m_core->m_allocate) VP_THROW(VERR_PARAMETER_IS_NOTALLOCATED);

	VTensorDict xs = { {"#", x} };
	VTensorDict ys = evaluateEx(train, xs);
	
	if (ys.size() != 1) VP_THROW(VERR_UNDEFINED);

	return ys["#"];
}

VTensorDict VModule::evaluateEx(bool train, VTensorDict xs) {
	int nDivisions = session().device_man().getUsingDeviceCount(getNth());
	m_core->m_splitDataIdx(nDivisions);

	VTensorDict preds;
	VExecTracer tracer;
	
	//printf("MP1: module#%d\n", getNth());
	VExecTracerPool tracers = m_core->m_tracerPools[train ? 0 : 1];
	//printf("MP2: module#%d\n", getNth());

	if (tracers.openAndExecute(xs, 10, preds, tracer)) {
		return preds;
	}
	//printf("MP3: module#%d\n", getNth());
	;
	int nInputDevice = xs.begin()->second.device();

	int64 batch_size = xs.begin()->second.shape()[0];

	bool noGrad = session().getNoGrad() || !train;
	
	if (0) printf("VModule::evaluateEx: train=%d, session().getNoGrad()=%d, noGrad=%d\n", train, session().getNoGrad(), noGrad);

	if (nInputDevice >= 0) { // 병렬 처리 중 커스텀 레이어 콜백 처리 중에 다시 호출되면서 이미 특정 스레드를 타고 온 상태
		// Gan 같은 경우 생성기 출력이 입력 디바이스여서 아래 줄 홧인사살 누락하면 판별기가 CPU 모드로 이상 동작할 수 있음
		session().device_man().setCurDevice(nInputDevice, tracer);
		VTensorDict sideTerms; // 콜백 처리 중 재호출시의 sideTerms 전달 여부는 고려하지 않은 상태임
		VTensorDict output = m_evaluate(m_core, xs, train, noGrad, nInputDevice, sideTerms, tracer);
		tracer.closeRecording(output);
		return output;
	}
	else if (nDivisions <= 0) { // 호스트메모리를 이용한 처리, 콜백 처리 중의 재호출일 수도 있다.
		VTensorDict sideTerms; // 콜백 처리 중 재호출시의 sideTerms 전달 여부는 고려하지 않은 상태임
		session().device_man().setCurDevice(-1, tracer);
		VTensorDict output = m_evaluate(m_core, xs, train, noGrad, -1, sideTerms, tracer);
		tracer.closeRecording(output);
		return output;
	}
	else {
		VList contexts;
		std::thread** ppThreads = new std::thread * [nDivisions];

		for (int n = 0; n < nDivisions; n++) {
			contexts.push_back(VDict());
			ppThreads[n] = NULL;
		}

		tracer.openBranch(nDivisions);

		for (int n = 0; n < nDivisions; n++) {
			int nDevice = session().device_man().getNthUsingDevice(getNth(), n);

			VTensorDict slice_xs;
			
			for (auto& it : xs) {
				VTensor x = it.second;
				VTensor sliceX = x.getNthSlice(nDivisions, n, nDevice, tracer);
				if (!sliceX.isValid()) continue;
				slice_xs[it.first] = sliceX;
			}

			if (slice_xs.size() == 0) {
				tracer.setVoidBranch(n);
				continue;
			}

			VExecTracer childTracer = tracer.setValidBranch(n, "forward_branch", slice_xs);
			VDict sHandles = vutils.toDictInternal(slice_xs);

			VDict ctx{ {"this", this->cloneCore()}, {"session", session().cloneCore()}, {"xs", sHandles}, {"train", train},
				{"device", nDevice}, {"no_grad", noGrad}, {"errcode", 0}, {"tracer", (VObjCore*)childTracer.getClone()} };

			contexts[n] = ctx;
		}

		tracer.setFork(nDivisions);

		for (int n = 0; n < nDivisions; n++) {
			VDict ctx = contexts[n];
			if (ctx.size() == 0) continue;
			ppThreads[n] = new std::thread(ms_evaluateMain, ctx.cloneCore());
		}

		string failReport;
		bool first = true;

		VTensorDict ys;

		for (int64 n = 0, slice_from = 0, slice_next = 0; n < nDivisions; n++, slice_from = slice_next) {
			if (ppThreads[n] == NULL) continue;

			ppThreads[n]->join();

			VDict ctx = contexts[n];

			int errcode = vutils.seek_dict(ctx, "errcode", 0);

			if (errcode != 0) {
				//failReport += "thread-" + to_string(n) + ": " + errcode + "\n";
				//VP_THROW1(VERR_FAIL_IN_PARRALLEL_EVALUATION, failReport);
				VException exInfo((VExceptionCore*)(VObjCore*)ctx["errinfo"]);
				VP_THROW1(VERR_PARALLEL_EVALUATION, exInfo);
			}

			VDict sHandles = ctx["ys"];
			VTensorDict slice_ys = vutils.toTensorDict(session(), sHandles);

			for (auto& it : slice_ys) {
				// bert 처리 중 출력 외에 원래 입력인 #x가 남아 있어 오류를 발생시켜 이를 걸러내기 위해 아래 줄을 추가했었음
				// 그러나 이름 붙지 않은 단일 출력을 구하는 다른 모델들에서는 #x가 출력에 해당해 걸러내서는 안됨
				// 따라서 일단 주석 처리를 해 놓고 bert 같은 경우에 어떻게 이를 걸러낼지 별도로 고민해보기로 한다.
				//if (it.first == "#") continue;

				VTensor y;
				VTensor sliceY = it.second;

				if (ys.find(it.first) == ys.end()) {
					VDataType type = sliceY.type();
					VShape shape = sliceY.shape().replace_nth(0, batch_size);

					y = tracer.createTensor(session(), shape, type, 0);

					ys[it.first] = y;
				}
				else {
					y = ys[it.first];
				}

				y.copySliceFrom(sliceY, slice_from, tracer);
				y.keepBackpropMergeInfo(*this, sliceY);

				slice_next = slice_from + sliceY.shape()[0];

			}

			ctx.freeClone();
		}

		delete[] ppThreads;

		if (failReport != "") VP_THROW1(VERR_PARALLEL_EVALUATION, failReport);

		tracer.closeRecording(ys);
		return ys;
	}
}

VList VModule::GetBuiltinCustomNames() {
	VList list;
	for (auto& it : VModuleCore::ms_builtinCustom) list.push_back(it);
	return list;
}

/*
VList VModule::GetBuiltinModelNames() {
	VList list;
	for (auto& it : VModuleCore::ms_builtinModel) list.push_back(it);
	return list;
}
*/

VList VModule::GetBuiltinLayerNames() {
	VList list;
	for (auto& it : VModuleCore::ms_builtinLayer) list.push_back(it);
	return list;
}

VList VModule::GetBuiltinNetworkNames() {
	VList list;
	for (auto& it : VModuleCore::ms_builtinNetwork) list.push_back(it);
	return list;
}

VTensorDict VModule::m_evaluate(VModuleCore* pStarter, VTensorDict xs, bool train, bool noGrad, int nDevice, VTensorDict& sideTerms, VExecTracer tracer) {
	return m_core->m_evaluate(pStarter, xs, train, noGrad, nDevice, sideTerms, tracer);
}

void VModule::ms_evaluateMain(void* aux) {
	VDict ctx((VDictCore*)aux);

	try {
		VSession session((VHSession)ctx["session"]);
		VModule module(session, (VHModule)ctx["this"]);

		VTensorDict xs = vutils.toTensorDict(session, (VDict)ctx["xs"]);
		VExecTracer tracer((VExecTracerCore*)(VObjCore*)ctx["tracer"]);

		bool train = ctx["train"];
		bool noGrad = ctx["no_grad"];
		int nDevice = ctx["device"];

		session.device_man().setCurDevice(nDevice, tracer);

		VTensorDict sideTerms;

		VTensorDict ys = module.m_evaluate(module.getCore(), xs, train, noGrad, nDevice, sideTerms, tracer);

		tracer.closeRecording(ys);

		ctx["ys"] = vutils.toDictInternal(ys);
	}
	//catch (ValException vex) { ctx["errcode"] = VERR_UNDEFINED;  ex= ctx["errinfo"] = ; }
	catch (ValException ex) {
		ctx["errcode"] = ex.m_nErrCode;
		VException ex1(ex.m_nErrCode, ex.m_file, ex.m_line);
		VException ex2(VERR_PARALLEL_EVALUATION, ex1, __FILE__, __LINE__);
		ctx["errinfo"] = (VObjCore*)ex2.cloneCore();
	}
	catch (VException ex) {
		ctx["errcode"] = ex.GetErrorCode();
		ctx["errinfo"] = (VObjCore*)ex.cloneCore();
	}
	catch (...) {
		ctx["errcode"] = VERR_PARALLEL_EVALUATION;
		VException ex(VERR_PARALLEL_EVALUATION, __FILE__, __LINE__);
		ctx["errinfo"] = (VObjCore*)ex.cloneCore();
	}
}

VGraph VModule::getGraphTemplate() {
	return m_core->m_graphTemplate;
}

//-----------------------------------------------------------------------------------------------------
// 코어 영역 확장 코드

void VModuleCore::m_onCreate() {
	m_sDevice = "cpu";
	m_bMacroExpanded = false;
	m_nParamSize = 0;
	m_bNeedForwardCallbeck = false;
	m_bNeedBackwardCallbeck = false;

	if (ms_inBuiltinLayerNames(m_sBuiltin)) {
		m_moduleType = VModuleType::layer;
		m_bIncludingMacro = false;
		//m_createModuleParam();
		//m_createLayerGraph();
	}
	else if (ms_inBuiltinNetworkNames(m_sBuiltin)) {
		m_moduleType = VModuleType::network;
		m_bIncludingMacro = false;
		//m_createNetworkGraph();
	}
	/*
	else if (ms_inBuiltinModelNames(m_sBuiltin)) {
		m_moduleType = VModuleType::model;
		m_bIncludingMacro = false;
		//m_createModel();
	}
	*/
	else if (ms_inBuiltinCustomNames(m_sBuiltin)) {
		m_moduleType = VModuleType::custom;
		m_bIncludingMacro = false;
	}
	else if (m_sBuiltin == "macro") {
		m_moduleType = VModuleType::macro;
		m_bIncludingMacro = true;
	}
	else if (m_sBuiltin == "user_defined") {
		m_moduleType = VModuleType::user_defined;
		m_bIncludingMacro = false;
	}
	else {
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}

	m_tracerPools[0] = VExecTracerPool(m_session, "");
	m_tracerPools[1] = VExecTracerPool(m_session, "");
}

void VModuleCore::m_onDelete() {
}

void VModuleCore::m_setup(VDict moduleInfo) {
	m_sName = "no_name";

	m_bMacroExpanded = moduleInfo["macro_expanded"];
	m_bIncludingMacro = moduleInfo["macro_included"];
	m_macroArgs = moduleInfo["macro_args"];

	m_inShape = moduleInfo["in_shape"];
	m_outShape = moduleInfo["out_shape"];

	m_setName = (string)moduleInfo["set_name"];
	m_getName = (string)moduleInfo["get_name"];

	m_nParamSize = moduleInfo["param_size"];

	VList param_info = moduleInfo["params"];

	for (auto& it : param_info) {
		VValue pm_info = m_loadParamInfo(it);
		m_params.push_back(pm_info);
		/*
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		*/
	}

	VList children_info = moduleInfo["children"];

	for (auto& it : children_info) {
		VDict childInfo = it;
		VModule childModule(m_session, childInfo);
		m_children.push_back(childModule);
		//VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		/*
		VModule childModule = m_loadModuleWithSerializeInfo(it);
		m_children.push_back(childModule);
		*/
	}
}

void VModuleCore::m_setup(string sFormula, VDict paramInfo) {
	m_formula = sFormula;
	m_formulaParamInfo = paramInfo;
}

void VModuleCore::m_openGraph(int depth, VShape& shape, VDict shapeDict, bool trace) {
	try {
		string getName = (string)vutils.seek_dict(m_propDict, "get", "");
		if (getName != "" && getName[0] != '#') {
			m_getName = getName;

			if (shape.size() > 0) {
				VShape getShape = vutils.seek_dict(shapeDict, m_getName, VShape());
				if (getShape.size() == 0) VP_THROW1(VERR_UNKNWON_SHAPE_FOR_GET_FIELD, m_getName);
				shape = getShape.insert_head(shape[0]);
				if (trace) printf("%*s[IN] get %s: input shape %s was loaded\n", depth, "", getName.c_str(), shape.desc().c_str());
			}
		}

		m_bMacroExpanded = true;
		m_shapeExpanded = shape.copy();

		if (shape.size() > 0) {
			m_inShape = shape.copy();
			if (trace) printf("%*s[IN] m_openGraph(shape:%s) for %s called\n", depth, "", shape.desc().c_str(), this->m_sBuiltin.c_str());
		}

		VShape yshape;
		int64 tsize;

		switch (m_moduleType) {
		case VModuleType::layer:
			m_createModuleParam(depth, shape, shapeDict);
			m_createLayerGraph();
			break;
		case VModuleType::network:
			if (m_sBuiltin == "sequential" || m_sBuiltin == "residual") {
				if (m_propDict.find("repeat") != m_propDict.end()) {
					if (m_propDict["repeat"].is_int()) {
						int repeat = m_propDict["repeat"];
						if (repeat > 1) {
							m_resolve_repeat(repeat);
						}
					}
				}
			}

			m_createNetworkGraph();

			if (m_sBuiltin == "stack") {
				tsize = vutils.seek_dict(m_propDict, "tail_size", 1);
				m_graphTemplate.setOption("tsize", tsize);
			}
			else if (m_sBuiltin == "squeezeexcitation") {
			}

			for (auto& it : m_children) {
				if (shape.size() > 0) {
					if (m_sBuiltin == "add" || m_sBuiltin == "parallel" || m_sBuiltin == "stack") shape = m_inShape.copy();
				}

				it.m_core->m_openGraph(depth + 1, shape, shapeDict, trace);
				m_params.push_back(it.params());

				if (shape.size() > 0) {
					if (m_sBuiltin == "parallel") {
						if (yshape.size() == 0) yshape = shape;
						else if (yshape.size() == 4) {
							if (shape.remove_nth(1) != yshape.remove_nth(1)) { VP_THROW(VERR_SIZE_GRAPH); }
							yshape = shape.replace_nth(1, shape[1] + yshape[1]);
						}
						else {
							if (shape.remove_end() != yshape.remove_end()) { VP_THROW(VERR_SIZE_GRAPH); }
							yshape = shape.replace_end(shape[-1] + yshape[-1]);
						}
					}
					else if (m_sBuiltin == "add") {
						if (yshape.size() == 0) yshape = shape;
						else if (shape != yshape) {
							printf("shape: %s vs %s\n", shape.desc().c_str(), yshape.desc().c_str());
							VP_THROW(VERR_SHAPE_GRAPH);
						}
					}
					else if (m_sBuiltin == "stack") {
						int64 batch_size = shape[0];
						int64 stack_size = shape.total_size() / (batch_size * tsize);
						if (shape.total_size() % (batch_size * tsize) != 0) { VP_THROW(VERR_SIZE_GRAPH); }
						if (yshape.size() == 0) yshape = { batch_size, stack_size, tsize };
						else {
							if (shape[0] != yshape[0]) { VP_THROW(VERR_SHAPE_GRAPH); }
							yshape = yshape.replace_nth(1, yshape[1] + stack_size);
						}
					}
					else if (m_sBuiltin == "squeezeexcitation") {
					}
				}
			}

			if (shape.size() > 0) {
				if (m_sBuiltin == "parallel") { shape = yshape; }
				else if (m_sBuiltin == "stack") { shape = yshape; }
				else if (m_sBuiltin == "pruning") { shape = m_inShape.copy(); }
				else if (m_sBuiltin == "squeezeexcitation") { shape = m_inShape.copy(); }
				else if (m_sBuiltin == "residual") {
					if (shape.size() == 4) {
						if (shape.remove_nth(1) != m_inShape.remove_nth(1)) VP_THROW(VERR_SHAPE_GRAPH);
						if (shape[1] % m_inShape[1] != 0) VP_THROW(VERR_SHAPE_GRAPH);
					}
					else {
						if (shape.remove_end() != m_inShape.remove_end()) VP_THROW(VERR_SHAPE_GRAPH);
						if (shape[-1] % m_inShape[-1] != 0) VP_THROW(VERR_SHAPE_GRAPH);
					}
				}
			}
			break;
		/*
		case VModuleType::model:
			m_createModel();
			for (auto& it : m_children) {
				it.m_core->m_openGraph(depth + 1, shape, shapeDict, trace);
				m_params.push_back(it.params());
			}
			break;
		*/
		case VModuleType::macro:
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
			break;
		case VModuleType::custom:
			for (auto& it : m_children) {
				it.m_core->m_openGraph(depth + 1, shape, shapeDict, trace);
				m_params.push_back(it.params());
			}
			break;
		case VModuleType::user_defined:
			m_createFormulaParam(depth, shape, shapeDict);
			//m_createFormulaGraph();
			m_createLayerGraph();
			//VP_THROW(VERR_NOT_IMPLEMENTED_YET);
			break;
		default:
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
			break;
		}

		string setName = vutils.seek_dict(m_propDict, "set", "");
		if (setName != "" && setName[0] != '#') {
			m_setName = setName;
			if (shape.size() > 0) {
				if (trace) printf("%*s[OUT] set %s: shape %s was saved\n", depth, "", setName.c_str(), shape.desc().c_str());
				shapeDict[setName] = shape.remove_head(); // shape.copy();
			}
		}

		if (shape.size() > 0) {
			m_outShape = shape.copy();
			if (trace) printf("%*s[OUT] m_openGraph(shape:%s) for %s ended\n", depth, "", shape.desc().c_str(), this->m_sBuiltin.c_str());
		}
	}
	catch (VException ex) {
		VP_THROW2(VERR_MODULE_EXPAND_FAILURE, ex, m_sBuiltin);
	}
	catch (...) {
		printf("m_sBuiltin = %s\n", m_sBuiltin.c_str());
		VP_THROW1(VERR_MODULE_EXPAND_FAILURE, m_sBuiltin);
	}
}

VModule VModuleCore::m_expandMacro(VShape& shape, VDict kwArgs) {
	VModule module(this);
	return module.expand(shape, kwArgs);
}

void VModuleCore::m_resolve_repeat(int repeat) {
	m_propDict.erase("repeat");

	VModuleList children;
	VModule me(this);

	for (int n = 0; n < repeat; n++) {
		VModule clone(me, true);
		children.push_back(clone);
	}

	m_sBuiltin = "sequential";
	m_children = children;
}

void VModuleCore::m_setup(VModuleCore* pSrcCore, bool copyChildren) {
	m_moduleType = pSrcCore->m_moduleType;
	m_sName = pSrcCore->m_sName;
	m_sMacroName = pSrcCore->m_sMacroName;
	m_sDevice = pSrcCore->m_sDevice;
	m_bMacroExpanded = pSrcCore->m_bMacroExpanded;
	m_bIncludingMacro = pSrcCore->m_bIncludingMacro;
	m_macroArgs = vutils.copy(pSrcCore->m_macroArgs);
	m_moduleType = pSrcCore->m_moduleType;
	m_allocate = pSrcCore->m_allocate;

	//m_params = vutils.copy(pSrcCore->m_params);
	m_nParamSize = 0; // pSrcCore->m_nParamSize;

	m_shapeExpanded = pSrcCore->m_shapeExpanded.copy();
	m_inShape = pSrcCore->m_inShape.copy();
	m_outShape = pSrcCore->m_outShape.copy();

	m_setName = pSrcCore->m_setName;
	m_getName = pSrcCore->m_getName;

	m_graphTemplate = pSrcCore->m_graphTemplate;

	m_bNeedForwardCallbeck = pSrcCore->m_bNeedForwardCallbeck;
	m_bNeedBackwardCallbeck = pSrcCore->m_bNeedBackwardCallbeck;

	for (auto& it : pSrcCore->m_cbForwardItemMap) {
		m_cbForwardItemMap[it.first] = it.second;
	}

	for (auto& it : pSrcCore->m_cbBackwardItemMap) {
		m_cbBackwardItemMap[it.first] = it.second;
	}

	m_formula = pSrcCore->m_formula;

	for (auto& it : pSrcCore->m_formulaParamInfo) {
		m_formulaParamInfo[it.first] = it.second;
	}

	// 아래 항들은 내용 없으면 새로 생성하므로 복사하지 않는다.
	// Parameters m_parameters;
	// map<int, VGraph> m_graphMap;
	// VList m_dataIdx;
	// VList m_dataIdxes;

	if (copyChildren) {
		for (auto& it : pSrcCore->m_children) {
			VModule clone(it, true);
			m_children.push_back(clone);
		}
	}
}

void VModuleCore::m_createModuleParam(int depth, VShape& shape, VDict shapeDict) {
	VDict param;

	if (m_sBuiltin == "linear") m_createLinearParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "dense") m_createDenseParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "addbias") m_createAddBiasParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "conv2d" || m_sBuiltin == "conv2d_transposed") m_createConv2dParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "conv2d_dilated") m_createConv2dDilatedParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "batchnorm") m_createBatchnormParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "rnn") m_createRnnParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "lstm") m_createRnnParam(depth, shape, shapeDict, param, "lstm");
	else if (m_sBuiltin == "gru") m_createRnnParam(depth, shape, shapeDict, param, "gru");
	else if (m_sBuiltin == "activate") m_createActivateParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "embed") m_createEmbedParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "mh_attention") m_createMultiHeadAttentionParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "max" || m_sBuiltin == "avg") m_createPoolParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "upsample") m_createUpsampleParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "reshape") m_createReshapeParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "transpose") m_createTransposeParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "dropout") m_createDropoutParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "extract") m_createExtractParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "layernorm") m_createLayernormParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "flatten") m_createFlattenParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "pass") m_createPassParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "globalavg") m_createGlobalAvgParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "adaptiveavg") m_createAdaptiveAvgParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "concat") m_createConcatParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "selectntop" || m_sBuiltin == "selectntoparg") m_createSelectNTopParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "normal_noise") m_createNormalNoiseParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "uniform_noise") m_createUniformNoiseParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "normal_random") m_createNormalRandomParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "uniform_random") m_createUniformRandomParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "round") m_createRoundParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "cosinesim") m_createCosineSimParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "codeconv") m_createCodeConvParam(depth, shape, shapeDict, param);
	else if (m_sBuiltin == "relu") m_createActivateParam(depth, shape, shapeDict, param, "relu");
	else if (m_sBuiltin == "gelu") m_createActivateParam(depth, shape, shapeDict, param, "gelu");
	else if (m_sBuiltin == "selu") m_createActivateParam(depth, shape, shapeDict, param, "selu");
	else if (m_sBuiltin == "tanh") m_createActivateParam(depth, shape, shapeDict, param, "tanh");
	else if (m_sBuiltin == "sigmoid") m_createActivateParam(depth, shape, shapeDict, param, "sigmoid");
	else if (m_sBuiltin == "mish") m_createActivateParam(depth, shape, shapeDict, param, "mish");
	else if (m_sBuiltin == "swish") m_createActivateParam(depth, shape, shapeDict, param, "swish");
	else if (m_sBuiltin == "leaky") m_createActivateParam(depth, shape, shapeDict, param, "leaky");
	else if (m_sBuiltin == "softmax") m_createActivateParam(depth, shape, shapeDict, param, "softmax");
	else if (m_sBuiltin == "random") {
		VP_THROW1(VERR_NOT_IMPLEMENTED_YET, m_sBuiltin);
	}
	else {
		VP_THROW1(VERR_NOT_IMPLEMENTED_YET, m_sBuiltin);
	}

	param["module_name"] = m_sName;
	m_params.push_back(param);
}

void VModuleCore::m_createFormulaParam(int depth, VShape& shape, VDict shapeDict) {
	VDict param;
	
	param["module_name"] = m_sName;

	m_nParamSize = 0;

	for (auto& it : m_formulaParamInfo) {
		VDict pmInfo = it.second;
		string type = vutils.seek_dict(pmInfo, "type", "");
		if (type == "param") {
			bool needGrad = vutils.seek_dict(pmInfo, "need_grad", true);

			VShape shape = pmInfo["shape"];

			int64 in_width = shape.total_size() / shape[-1];
			int64 out_width = shape[-1];;

			string init_method = vutils.seek_dict(pmInfo, "init_method", "xavier");
			float init_arg = ms_getInitArg(init_method, in_width, out_width, pmInfo);

			if (m_allocate) VOptimizer::createParam(m_session, it.first, shape, needGrad, init_method, init_arg, param);

			m_nParamSize += shape.total_size();
		}
		else if (type == "param_empty") {
			if (m_allocate) VOptimizer::createEmptyParam(m_session, it.first, param);
		}
		else if (type == "int") {
			param[it.first] = (int)pmInfo["value"];
		}
		else if (type == "float") {
			param[it.first] = (float)pmInfo["value"];
		}
		else if (type == "shape") {
			param[it.first] = pmInfo["value"];
		}
		else {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
	}

	m_params.push_back(param);
}

float VModuleCore::ms_getInitArg(string init_method, int64 in_width, int64 out_width, VDict args, float def) {
	if (init_method == "xavier") return ::sqrtf(1.0f / (float)in_width);
	if (init_method == "normalized_xavier") return ::sqrtf(6.0f / (float)(in_width+out_width));
	if (init_method == "he") return ::sqrtf(2.0f / (float)in_width);
	
	return vutils.seek_dict(args, "init_arg", def);
}

void VModuleCore::m_createLinearParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int64 inWidth = vutils.seek_dict(m_propDict, "in_width", (shape.size() > 0) ? shape[-1] : 0);
	int64 outWidth = (m_propDict.find("out_width") == m_propDict.end()) ? m_propDict["width"] : m_propDict["out_width"];

	bool add_bias = vutils.seek_dict(m_propDict, "add_bias", true);

	//float range = ::sqrtf((float)inWidth); // mistake? ::sqrtf(1.0f / (float)inWidth);

	string init_method = vutils.seek_dict(m_propDict, "init_method", "xavier");
	float init_arg = ms_getInitArg(init_method, inWidth, outWidth, m_propDict);

	if (inWidth == 0) VP_THROW(VERR_INVALID_DICT_KEY);

	VShape wshape = VShape{ outWidth, inWidth };
	if (m_allocate) VOptimizer::createAffineParam(m_session, wshape, add_bias, init_method, init_arg, param);
	//VOptimizer::createAffineParam(m_session, wshape, add_bias, "gauss", 0.03f, param);
	param["type"] = "linear";

	if (shape.size() > 0) shape[-1] = outWidth;

	m_nParamSize = inWidth * outWidth + outWidth;
}

void VModuleCore::m_createDenseParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int64 inWidth = vutils.seek_dict(m_propDict, "in_width", (shape.size() > 0) ? shape[-1] : 0);
	int64 outWidth = (m_propDict.find("out_width") == m_propDict.end()) ? m_propDict["width"] : m_propDict["out_width"];

	if (inWidth == 0) VP_THROW(VERR_UNDEFINED);

	bool add_bias = vutils.seek_dict(m_propDict, "add_bias", true);
	string actfunc = vutils.seek_dict(m_propDict, "actfunc", "relu");

	string default_init_method = (actfunc == "relu") ? "he" : "xavier";
	string init_method = vutils.seek_dict(m_propDict, "init_method", default_init_method);

	float init_arg = ms_getInitArg(init_method, inWidth, outWidth, m_propDict);

	VShape wshape = VShape{ outWidth, inWidth };

	if (m_allocate) VOptimizer::createAffineParam(m_session, wshape, add_bias, init_method, init_arg, param);

	param["type"] = "dense";
	param["actfunc"] = (int)VConsts::getActFunc(actfunc);
	param["leaky_alpha"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "leaky_alpha", 0.1f));

	if (shape.size() > 0) shape[-1] = outWidth;

	m_nParamSize = inWidth * outWidth;
	if (add_bias) m_nParamSize += outWidth;
}

void VModuleCore::m_createAddBiasParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int64 width = vutils.seek_dict(m_propDict, "width", (shape.size() > 0) ? shape[-1] : 0);

	if (width == 0) VP_THROW(VERR_UNDEFINED);

	string init_method = vutils.seek_dict(m_propDict, "init_method", "xavier");
	float init_arg = ms_getInitArg(init_method, width, width, m_propDict);

	VShape bshape = VShape{ width };

	if (m_allocate) VOptimizer::createBiasParam(m_session, bshape, init_method, init_arg, param);
	param["type"] = "bias";

	if (shape.size() > 0) shape[-1] = width;

	m_nParamSize = width;
}

VShape VModuleCore::m_get2dArg(string plainKey, string elemPrefix, string shapeName, int64 nDef1, int64 nDef2) {
	if (m_propDict.find(shapeName) != m_propDict.end()) {
		VShape shape = m_propDict[shapeName];
		if (shape.size() != 2) VP_THROW1(VERR_SHAPE_NOT_2D_DIMENSION, shapeName);
		return shape;
	}

	int64 plain = vutils.seek_dict(m_propDict, plainKey, 0);
	int64 x = vutils.seek_dict(m_propDict, elemPrefix + "x", plain ? plain : nDef1);
	int64 y = vutils.seek_dict(m_propDict, elemPrefix + "y", plain ? plain : nDef2 > 0 ? nDef2 : nDef1);

	return VShape{ x, y };
}

void VModuleCore::m_getPaddingArg(VDict param, VShape& shape, VShape ksize, VShape stride) {
	// padding(int, tuple or str, optional) – Padding added to all four sides of the input.Default: 0
	// padding_mode(str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'.Default : 'zeros'
	
	VShape pshape = VShape{ (ksize[0] - 1) / 2, ksize[0] / 2, (ksize[1] - 1) / 2, ksize[1] / 2 };

	if (stride.total_size() == 1 && m_propDict.find("padding") != m_propDict.end()) {
		VValue padding = m_propDict["padding"];

		if (m_sBuiltin != "conv2d") {
			// maxpool, avgpool: 정수 혹은 (정수,정수) 형태만 처리할 것
			if (!padding.is_string() || (string)padding != "same") VP_THROW1(VERR_WILL_BE_IMPLEMENTED, "padding for " + m_sBuiltin);
		}

		if (padding.is_int()) {
			int64 psize = padding;
			pshape = VShape{ psize, psize, psize, psize };
		}
		else if (padding.is_shape()) {
			pshape = padding;
			if (pshape.size() != 4) VP_THROW(VERR_BAD_PADDING_DIMENSION);
		}
		else if (padding.is_string()) {
			string sPadding = padding;
			if (sPadding == "valid") {
				pshape = VShape{ 0, 0, 0, 0 };
			}
			else if (sPadding != "same") VP_THROW1(VERR_BAD_PADDING_ARGUMENT, sPadding);
		}
		else {
			VP_THROW1(VERR_BAD_PADDING_ARGUMENT, padding.desc());
		}
	}

	if (shape.size() > 0) {
		if (shape.size() != 4) VP_THROW2(VERR_BAD_SHAPE_FOR_IMAGE_LAYER, m_sBuiltin, shape.desc());

		shape[2] += pshape[0] + pshape[1] + 1 - ksize[0];
		shape[3] += pshape[2] + pshape[3] + 1 - ksize[1];
	}

	param["padding"] = pshape;
	param["padding_mode"] = vutils.toPaddingMode(vutils.seek_dict(m_propDict, "padding_mode", "zeros"));
}

void VModuleCore::m_createConv2dParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	VShape ksize = m_get2dArg("ksize", "k", "kshape", 3);

	int64 xchn = vutils.seek_dict(m_propDict, "xchn", (shape.size() > 0) ? shape[1] : 0);
	int64 ychn = (m_propDict.find("ychn") == m_propDict.end()) ? m_propDict["chn"] : m_propDict["ychn"];

	string actfunc = vutils.seek_dict(m_propDict, "actfunc", "relu");

	param["actfunc"] = (int)VConsts::getActFunc(actfunc);
	param["leaky_alpha"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "leaky_alpha", 0.1f));
	
	if (xchn == 0) VP_THROW(VERR_UNDEFINED);

	int64 group = vutils.seek_dict(m_propDict, "group", -1);

	if (group >= 0) {
		if (group == 0) group = xchn;
		if (xchn % group != 0) VP_THROW2(VERR_BAD_GROUP_IN_CONV2D_LAYER, to_string(xchn), to_string(group));
		xchn = xchn / group;
	}

	string default_init_method = (actfunc == "relu") ? "he" : "xavier";
	string init_method = vutils.seek_dict(m_propDict, "init_method", default_init_method);

	float init_arg = ms_getInitArg(init_method, xchn * ksize[0] * ksize[1], ychn, m_propDict);

	VShape kshape{ ychn, xchn, ksize[0], ksize[1] };

	// batchnorm 사용하는 경우 bias 대신 shift 사용하려 조정 중, 효과 확인시 유사 클래스에 전파 필요
	//bool batchnorm = vutils.seek_dict(m_propDict, "batchnorm", false);
	//bool add_bias = !batchnorm && vutils.seek_dict(m_propDict, "add_bias", true);
	bool add_bias = vutils.seek_dict(m_propDict, "add_bias", true);

	if (m_allocate) VOptimizer::createAffineParam(m_session, kshape, add_bias, init_method, init_arg, param);

	param["type"] = "conv2d";

	VShape stride = m_get2dArg("stride", "s", "sshape", 1);

	param["stride"] = stride;

	int64 sx = stride[0];
	int64 sy = stride[1];

	if (shape.size() > 0) {
		shape[1] = ychn;

		if (m_sBuiltin == "conv2d") {
			if (sx != 1 || sy != 1) shape = VShape{ shape[0], shape[1] , shape[2] / sx, shape[3] / sy };
		}
		else if (m_sBuiltin == "conv2d_transposed") {
			if (sx != 1 || sy != 1) shape = VShape{ shape[0], shape[1], shape[2] * sx, shape[3] * sy };
		}
		else {
			VP_THROW(VERR_INVALID_BUILTIN_CONV);
		}
	}

	m_nParamSize = kshape.total_size();

	if (add_bias) m_nParamSize += ychn;

	m_getPaddingArg(param, shape, ksize, stride);

	/*
	param["batchnorm"] = batchnorm;

	if (batchnorm) {
		bool rescale = vutils.seek_dict(m_propDict, "rescale", true);
		//bool shift = rescale && (bool)vutils.seek_dict(m_propDict, "shift", false);
		bool shift = rescale && (bool)vutils.seek_dict(m_propDict, "shift", true);

		if (m_allocate) VOptimizer::createBatchNormalParam(m_session, { ychn }, rescale, shift, param);

		param["momentum"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "momentum", 0.99f));
		param["epsilon"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "epsilon", 0.001f));

		m_nParamSize += ychn * (rescale ? (shift ? 4 : 3) : 2);
	}
	else {
		if (m_allocate) VOptimizer::createBatchNormalParam(m_session, {}, false, false, param);
	}
	*/
}

void VModuleCore::m_createConv2dDilatedParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	VShape ksize = m_get2dArg("ksize", "k", "kshape", 3);

	int64 xchn = vutils.seek_dict(m_propDict, "xchn", (shape.size() > 0) ? shape[1] : 0);
	int64 ychn = (m_propDict.find("ychn") == m_propDict.end()) ? m_propDict["chn"] : m_propDict["ychn"];

	string actfunc = vutils.seek_dict(m_propDict, "actfunc", "relu");

	param["actfunc"] = (int)VConsts::getActFunc(actfunc);
	param["leaky_alpha"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "leaky_alpha", 0.1f));

	string default_init_method = (actfunc == "relu") ? "he" : "xavier";
	string init_method = vutils.seek_dict(m_propDict, "init_method", default_init_method);

	float init_arg = ms_getInitArg(init_method, xchn * ksize[0] * ksize[1], ychn, m_propDict);

	if (xchn == 0) VP_THROW(VERR_INVALID_DICT_KEY);

	VShape kshape{ ychn, xchn, ksize[0], ksize[1] };

	bool add_bias = vutils.seek_dict(m_propDict, "add_bias", true);

	if (m_allocate) VOptimizer::createAffineParam(m_session, kshape, add_bias, init_method, init_arg, param);

	param["type"] = "conv2d_dilated";

	int64 stride = (int64)vutils.seek_dict(m_propDict, "stride", 1);
	int64 sx = vutils.seek_dict(m_propDict, "sx", stride);
	int64 sy = vutils.seek_dict(m_propDict, "sy", stride);

	param["stride"] = VShape({ sx, sy });

	int64 gap = vutils.seek_dict(m_propDict, "gap", 2);
	int64 gx = vutils.seek_dict(m_propDict, "gx", gap);
	int64 gy = vutils.seek_dict(m_propDict, "gy", gap);

	param["gap"] = VShape({ gx, gy });

	if (shape.size() > 0) {
		shape[1] = ychn;
		if (sx != 1 || sy != 1) shape = VShape{ shape[0], shape[1], shape[2] / sx, shape[3] / sy };
	}

	m_nParamSize = kshape.total_size();

	if (add_bias) m_nParamSize += ychn;

	/*
	bool batchnorm = vutils.seek_dict(m_propDict, "batchnorm", false);

	param["batchnorm"] = batchnorm;

	if (batchnorm) {
		bool rescale = vutils.seek_dict(m_propDict, "rescale", true);
		bool shift = rescale && (bool)vutils.seek_dict(m_propDict, "shift", false);

		if (m_allocate) VOptimizer::createBatchNormalParam(m_session, { ychn }, rescale, shift, param);

		param["momentum"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "momentum", 0.99f));
		param["epsilon"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "epsilon", 0.001f));

		m_nParamSize += ychn * (rescale ? (shift ? 4 : 3) : 2);
	}
	else {
		if (m_allocate) VOptimizer::createBatchNormalParam(m_session, {}, false, false, param);
	}
	*/
}

void VModuleCore::m_createBatchnormParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	bool rescale = vutils.seek_dict(m_propDict, "rescale", true);
	bool shift = vutils.seek_dict(m_propDict, "shift", true);

	int64 output_width = (shape.size() > 0) ? ((shape.size() == 4) ? shape[1] : shape[-1]) : (int64)m_propDict["out_width"];
	VShape bshape{ output_width };

	if (m_allocate) VOptimizer::createBatchNormalParam(m_session, bshape, rescale, shift, param);

	param["type"] = "batchnorm";
	param["momentum"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "momentum", 0.99f));
	param["epsilon"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "epsilon", 0.001f));

	m_nParamSize = output_width * (rescale ? (shift ? 4 : 3) : 2);
}

void VModuleCore::m_createRnnParam(int depth, VShape& shape, VDict shapeDict, VDict param, string cell) {
	if (cell == "") {
		cell = (string)vutils.seek_dict(m_propDict, "cell", "basic");
		if (cell == "") cell = "basic";
		if (cell == "lstm") m_sBuiltin = "lstm";
		else if (cell == "gru") m_sBuiltin = "gru";
	}

	int nGates = 1;
	if (cell == "lstm") nGates = 4;
	else if (cell == "gru") nGates = 3;

	int64 nInputSize = vutils.seek_dict(m_propDict, "in_width", (shape.size() > 0) ? shape[-1] : 0);
	int64 nRecurSize = m_propDict["out_width"];

	if (nInputSize == 0) VP_THROW(VERR_INVALID_DICT_KEY);

	bool use_bias = (bool)vutils.seek_dict(m_propDict, "use_bias", true);
	bool batch_first = (bool)vutils.seek_dict(m_propDict, "batch_first", false);
	bool bi_direct = (bool)vutils.seek_dict(m_propDict, "bidirectional", false);

	int64 nLayers = vutils.seek_dict(m_propDict, "num_layers", 1);
	int64 nStack = nLayers * (bi_direct ? 2 : 1);

	if (m_allocate) VOptimizer::createRnnParam(m_session, nGates, nRecurSize, nInputSize, nLayers, bi_direct, use_bias, param);

	param["type"] = cell;
	param["use_bias"] = use_bias;
	param["rec_size"] = nRecurSize;
	param["num_layers"] = nLayers;
	param["bidirectional"] = bi_direct;
	param["batch_first"] = batch_first;
	param["in_seq"] = (bool)vutils.seek_dict(m_propDict, "in_seq", true);
	param["out_seq"] = (bool)vutils.seek_dict(m_propDict, "out_seq", true);
	param["timesteps"] = (int64)vutils.seek_dict(m_propDict, "timesteps", 0);

	float keep_ratio = vutils.seek_dict(m_propDict, "keep_ratio", 1.0f);
	float drop_ratio = vutils.seek_dict(m_propDict, "drop_ratio", 1.0f - keep_ratio);

	param["drop_ratio"] = drop_ratio > 0 ? HYPER_REGVAL_CORE(drop_ratio) : -1;

	if (cell == "basic" || cell == "rnn") {
		string actfunc = vutils.seek_dict(m_propDict, "actfunc", "tanh");
		param["actfunc"] = (int)VConsts::getActFunc(actfunc);
		param["leaky_alpha"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "leaky_alpha", 0.1f));
	}

	if (cell == "lstm") {
		param["use_state"] = (bool)vutils.seek_dict(m_propDict, "use_state", false);
	}

	if (shape.size() > 0) {
		int64 time_axis = batch_first ? 1 : 0;
		if (!(bool)param["out_seq"]) shape = shape.replace_nth(time_axis, nStack);
		else if (!(bool)param["in_seq"]) shape = shape.insert_nth(time_axis, (int64)param["timesteps"]);
		
		shape = shape.replace_end(nRecurSize);
	}

	m_nParamSize = nGates * (nInputSize + nRecurSize) * nRecurSize;
	if (use_bias) m_nParamSize += nGates * nRecurSize * 2;
	m_nParamSize *= nStack;
}

void VModuleCore::m_createActivateParam(int depth, VShape& shape, VDict shapeDict, VDict param, string actfunc) {
	if (actfunc == "") {
		actfunc = (string)vutils.seek_dict(m_propDict, "actfunc", "relu");
	}

	param["type"] = "args";
	param["actfunc"] = (int)VConsts::getActFunc(actfunc);
	param["leaky_alpha"] = HYPER_REGVAL_CORE(vutils.seek_dict(m_propDict, "leaky_alpha", 0.1f));

}

void VModuleCore::m_createEmbedParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int64 vec_size = m_propDict["vec_size"];
	int64 voc_size = m_propDict["voc_size"];

	string init_method = vutils.seek_dict(m_propDict, "init_method", "gauss");
	float init_arg = ms_getInitArg(init_method, vec_size, vec_size, m_propDict, 1.0f);

	VShape wshape = VShape{ voc_size, vec_size };

	if (m_allocate) VOptimizer::createAffineParam(m_session, wshape, false, init_method, init_arg, param);

	param["type"] = "embed";
	param["position"] = (bool)vutils.seek_dict(m_propDict, "position", false);
	param["ndim"] = (int)vutils.seek_dict(m_propDict, "ndim", -1);

	if (shape.size() > 0) {
		if ((bool)param["position"]) {
			int ndim = vutils.seek_dict(m_propDict, "ndim", -1);
			if (ndim >= 0) {
				shape = shape.cut_tail(ndim);
			}
		}

		shape = shape.append(vec_size);
	}

	m_nParamSize = wshape.total_size();
}

void VModuleCore::m_createMultiHeadAttentionParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int64 vec_size = m_propDict["vec_size"];
	int64 head_cnt = m_propDict["head_cnt"];

	if (vec_size % head_cnt != 0) VP_THROW(VERR_BAD_HEAD_CNT_FOR_MH_ATTENTION);

	float coef = 1.0f / ::sqrtf(float(vec_size / head_cnt));

	string init_method = vutils.seek_dict(m_propDict, "init_method", "xavier");
	float init_arg = ms_getInitArg(init_method, vec_size / head_cnt, vec_size / head_cnt, m_propDict);

	if (m_allocate) VOptimizer::createAffineParam(m_session, VShape{ vec_size, vec_size }, true, init_method, init_arg, param, "K");
	if (m_allocate) VOptimizer::createAffineParam(m_session, VShape{ vec_size, vec_size }, true, init_method, init_arg, param, "Q");
	if (m_allocate) VOptimizer::createAffineParam(m_session, VShape{ vec_size, vec_size }, true, init_method, init_arg, param, "V");
	if (m_allocate) VOptimizer::createAffineParam(m_session, VShape{ vec_size, vec_size }, true, init_method, init_arg, param, "O");

	param["type"] = "mh_attention";
	param["head_cnt"] = head_cnt;
	param["coef"] = HYPER_REGVAL_CORE(coef);
	param["mask"] = vutils.seek_dict(m_propDict, "use_mask", false);

	int64 batch_size = shape[0];

	string key = vutils.seek_dict(m_propDict, "key", "");
	string query = vutils.seek_dict(m_propDict, "query", "");
	string value = vutils.seek_dict(m_propDict, "value", "");

	VShape kshape = (key != "") ? ((VShape)shapeDict[key]).insert_head(batch_size) : shape;
	VShape qshape = (query != "") ? ((VShape)shapeDict[query]).insert_head(batch_size) : shape;
	VShape vshape = (value != "") ? ((VShape)shapeDict[value]).insert_head(batch_size) : shape;

	if (kshape != qshape || kshape != vshape) {
		printf("kshape: %s, qshape:%s, vshape:%s\n", kshape.desc().c_str(), qshape.desc().c_str(), vshape.desc().c_str());
		VP_THROW(VERR_BAD_KQV_SHAPE_FOR_MH_ATTENTION);
	}

	param["key"] = key;
	param["query"] = query;
	param["value"] = value;

	m_nParamSize = (vec_size + 1) * vec_size * 4;

	shape = vshape;
}

void VModuleCore::m_createPoolParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	param["type"] = "pool";
	
	VShape stride = m_get2dArg("stride", "s", "sshape", 1);
	VShape ksize = m_get2dArg("ksize", "k", "kshape", stride[0], stride[1]);

	param["kernel"] = ksize;
	param["stride"] = stride;

	if (shape.size() > 0) {
		if (stride.total_size() != 1) shape = VShape{ shape[0], shape[1], shape[2] / stride[0], shape[3] / stride[1] };
	}

	m_getPaddingArg(param, shape, ksize, stride);
}

void VModuleCore::m_createUpsampleParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	param["type"] = "upsample";

	VShape stride = m_get2dArg("stride", "s", "sshape", 1);

	param["stride"] = stride;

	if (shape.size() > 0) {
		shape = VShape{ shape[0], shape[1], shape[2] * stride[0], shape[3] * stride[1] };
	}
}

void VModuleCore::m_createReshapeParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	VShape temp = m_propDict["shape"];
	VShape rshape = temp;

	if (shape.size() > 0) {
		if (rshape.total_size() < 0) {
			rshape = rshape.resolve_plcaeholder(shape.total_size());	// -1 처리 필요
			shape = rshape;
		}
		else {
			shape = shape.remove_tail_by_size(rshape.total_size()).append(rshape);
		}
	}

	param["type"] = "args";
	param["shape"] = rshape.copy();
}

void VModuleCore::m_createTransposeParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	VList axes = (VList)m_propDict["axes"];

	param["type"] = "args";
	param["axes"] = axes;

	if (shape.size() > 0) {
		shape = shape.transpose(axes);
	}
}

void VModuleCore::m_createDropoutParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	float keep_ratio = vutils.seek_dict(m_propDict, "keep_ratio", 1.0f);
	float drop_ratio = vutils.seek_dict(m_propDict, "drop_ratio", 1.0f - keep_ratio);

	param["type"] = "args";
	param["drop_ratio"] = HYPER_REGVAL_CORE(drop_ratio);
}

void VModuleCore::m_createExtractParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int64 axis = vutils.seek_dict(m_propDict, "axis", 0);
	int64 index = vutils.seek_dict(m_propDict, "index", 0);
	int64 count = vutils.seek_dict(m_propDict, "count", 1);
	bool reduce_axis = vutils.seek_dict(m_propDict, "reduce_axis", true);

	param["type"] = "args";
	param["axis"] = axis;
	param["index"] = index;
	param["count"] = count;
	param["reduce_axis"] = reduce_axis;

	shape[axis] = count;
	if (reduce_axis && count == 1) shape = shape.remove_nth(axis);
}

void VModuleCore::m_createLayernormParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int64 axis = vutils.seek_dict(m_propDict, "axis", 0);
	float scale = vutils.seek_dict(m_propDict, "scale", 1.0f);

	param["type"] = "args";
	param["axis"] = axis;
	param["scale"] = HYPER_REGVAL_CORE(scale);
}

void VModuleCore::m_createFlattenParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	if (shape.size() > 0) {
		shape = VShape{ shape[0], shape.total_size() / shape[0] };
	}
}

void VModuleCore::m_createPassParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	string dump_title = vutils.seek_dict(m_propDict, "dump", "");
	string direction = vutils.seek_dict(m_propDict, "direction", "");
	bool exception = vutils.seek_dict(m_propDict, "exception", false);

	param["type"] = "pass";
	param["dump"] = dump_title;
	param["exception"] = exception;
	param["direction"] = direction;
}

void VModuleCore::m_createGlobalAvgParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	if (shape.size() > 0) {
		shape = VShape{ shape[0], shape[1] };
	}
}

void VModuleCore::m_createAdaptiveAvgParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int64 size = vutils.seek_dict(m_propDict, "size", 1);

	int64 sx = vutils.seek_dict(m_propDict, "sx", size);
	int64 sy = vutils.seek_dict(m_propDict, "sy", size);

	param["size"] = VShape{ sx, sy };

	if (shape.size() > 0) {
		if (shape.size() != 4) VP_THROW(VERR_UNDEFINED);
		shape = shape.copy();
		shape[2] = size;
		shape[3] = size;
	}
}

void VModuleCore::m_createConcatParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	string bname = m_propDict["branch"];
	VShape bshape = shapeDict[bname];
	// bshape는 shape와 달리 배치크기 N 축을 포함하고 있지 않음에 유의
	if (shape.size() == 4) {
		if (shape.remove_head().remove_nth(0) != bshape.remove_nth(0)) {
			if (1) printf("m_createConcatParam: shape %s vs. bshape %s\n", shape.desc().c_str(), bshape.desc().c_str());
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
		shape = shape.replace_nth(1, shape[1] + bshape[0]);
	}
	else {
		if (shape.remove_head().remove_end() != bshape.remove_end()) {
			if (1) printf("m_createConcatParam: shape %s vs. bshape %s\n", shape.desc().c_str(), bshape.desc().c_str());
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
		shape = shape.replace_end(shape[-1] + bshape[-1]);
	}

	param["type"] = "args";
	param["branch"] = bname;
}

void VModuleCore::m_createSelectNTopParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int ntop = vutils.seek_dict(m_propDict, "ntop", 5);

	param["ntop"] = ntop;
}

void VModuleCore::m_createNormalNoiseParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	float mean = vutils.seek_dict(m_propDict, "mean", 0.0f);
	float std = vutils.seek_dict(m_propDict, "std", 1.0f);

	param["mean"] = HYPER_REGVAL_CORE(mean);
	param["std"] = HYPER_REGVAL_CORE(std);
}

void VModuleCore::m_createUniformNoiseParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	float min = vutils.seek_dict(m_propDict, "min", 0.0f);
	float max = vutils.seek_dict(m_propDict, "max", 1.0f);

	param["min"] = HYPER_REGVAL_CORE(min);
	param["max"] = HYPER_REGVAL_CORE(max);
}

void VModuleCore::m_createNormalRandomParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	VShape rshape = vutils.seek_dict(m_propDict, "shape", shape.remove_head());

	float mean = vutils.seek_dict(m_propDict, "mean", 0.0f);
	float std = vutils.seek_dict(m_propDict, "std", 1.0f);

	param["shape"] = rshape.copy();
	param["mean"] = HYPER_REGVAL_CORE(mean);
	param["std"] = HYPER_REGVAL_CORE(std);

	shape = rshape.insert_head(shape[0]);
}

void VModuleCore::m_createUniformRandomParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	VShape rshape = vutils.seek_dict(m_propDict, "shape", shape.remove_head());

	float min = vutils.seek_dict(m_propDict, "min", 0.0f);
	float max = vutils.seek_dict(m_propDict, "max", 1.0f);

	param["shape"] = rshape.copy();
	param["min"] = HYPER_REGVAL_CORE(min);
	param["max"] = HYPER_REGVAL_CORE(max);

	shape = rshape.insert_head(shape[0]);
}

void VModuleCore::m_createRoundParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	int prec = vutils.seek_dict(m_propDict, "prec", 1);

	param["prec"] = prec;
}

void VModuleCore::m_createCosineSimParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	string with_name = m_propDict["with"];

	param["with"] = with_name;
}

void VModuleCore::m_createCodeConvParam(int depth, VShape& shape, VDict shapeDict, VDict param) {
	string src_value = vutils.seek_dict(m_propDict, "src_value", "binary");
	string src_form = vutils.seek_dict(m_propDict, "src_form", "vector");
	string dest_value = vutils.seek_dict(m_propDict, "dest_value", "int");
	string dest_form = vutils.seek_dict(m_propDict, "dest_form", "scalar");

	if (src_value != "binary" || src_form != "vector") VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	if (dest_value != "int" || dest_form != "scalar") VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

void VModuleCore::m_createLayerGraph() {
	m_graphTemplate = VGraph(m_session, this, GraphInit::layer, m_session.device_man(), m_sBuiltin, m_params, m_propDict);
}

void VModuleCore::m_createNetworkGraph() {
	m_graphTemplate = VGraph(m_session, m_sBuiltin, m_session.device_man(), m_propDict);
}

VList VModuleCore::getDataIdx(int nDevice) {
	int nDivisions = m_session.device_man().getUsingDeviceCount(getNth());

	if (nDivisions <= 1 || m_dataIdx.size() == 0) return m_dataIdx;

	return m_dataIdxes[nDevice];
}

void VModuleCore::parallel_backprop(VTensor ygrad, VTensorList operands, VBackQueue* pQueue, VExecTracer tracer) {
	// 일단 cuda:0 지원만 가능한 형태로 임시 구현하되 crossRiver 일괄처리가 되는지만 확인한다.
	// 추후 비교적 간단한 모델을 이용하여 순전파에서의 병렬 처리 분기, 모음을 본격 구현하기로 한다.
	for (auto& it : operands) {
		pQueue->pushCross(it, ygrad); // 이것도 확인필요 multiple 변수이면 multiple grad가 당연
		// 즉 온전하려면 operands는 단수이어야...
		// 도강 인자 등록 메소드를 별도로 만들어 텐서별 정보관리는 폐기하기로 한다.
	}

	return;

	VP_THROW(VERR_NOT_IMPLEMENTED_YET);

	/*
	int nDivisions = (int)operands.size(); // 크기 0 배제로 인해 가용 디바이스 수와 다를 수 있음

	std::thread** ppThreads = new std::thread * [nDivisions];
	VList contexts;

	for (int n = 0; n < nDivisions; n++) {
		VTensor ygrad_slice = ygrad.getSlice(nDivisions, n, tracer);

		// (VHModule)(*this)를 (VHModule)this, (VHModule)m_core 등으로 잘못 처리하면 참조계수 증가가 누락되어 메모리 오류발생 원인이 된다.
		VDict ctx{ {"this", this->cloneCore()},  {"session", (VHSession)session()},  {"ygrad", ygrad_slice.cloneCore()},
				   {"tensor", operands[n].cloneCore()}, {"device", operands[n].device() } };
		contexts.push_back(ctx);

		ppThreads[n] = new std::thread(ms_backwardMain, ctx.cloneCore());
	}

	string failReport;

	for (int n = 0; n < nDivisions; n++) {
		ppThreads[n]->join();

		VDict ctx = contexts[n];

		string errcode = vutils.seek_dict(ctx, "errcode", "");

		if (errcode != "") {
			failReport += "thread-" + to_string(n) + ": " + errcode + "\n";
		}

		ctx.freeClone();
	}

	delete[] ppThreads;

	if (failReport != "") VP_THROW1(VERR_FAIL_IN_PARRALLEL_BACKWARD, failReport);
	*/
}

/*
void VModuleCore::m_createModel() {
	if (m_sBuiltin == "bert") {
		VModule bert(m_session, "sequential", m_propDict);
		bert.m_core->m_createBertModel();
		m_appendChild(bert);
	}
	else if (m_sBuiltin == "transformer_encoder") {
		VModule encoder(m_session, "sequential", m_propDict);
		encoder.m_core->m_createTransformerEncoderModel();
		m_appendChild(encoder);
	}
	else if (m_sBuiltin.substr(0, m_sBuiltin.length()-1) == "efficientnet-b") {
		char model_num = m_sBuiltin[m_sBuiltin.length() - 1];
		if (model_num >= '0' && model_num <= '7') {
			VModule efficientnet_template(m_session, "sequential", m_propDict);
			efficientnet_template.m_core->m_createEfficientNetModel(model_num);
			m_appendChild(efficientnet_template);
		}
		else {
			VP_THROW1(VERR_NOT_IMPLEMENTED_YET, m_sBuiltin);
		}
	}
	else {
		VP_THROW1(VERR_NOT_IMPLEMENTED_YET, m_sBuiltin);
	}
}

void VModuleCore::m_createBertModel() {
	int64 voc_size = m_propDict["voc_size"];
	int64 max_position = m_propDict["max_position"];
	int64 vec_size = vutils.seek_dict(m_propDict, "vec_size", 768);
	string sent_logit_name = vutils.seek_dict(m_propDict, "sent_logit_name", "sent_logits");
	string word_logit_name = vutils.seek_dict(m_propDict, "word_logit_name", "word_logits");

	VModule embed1(m_session, "add", {});

	VDict kwArdg11 = { {"idx", 0}, {"vec_size", vec_size}, {"voc_size", voc_size} };
	VDict kwArdg12 = { {"idx", 1}, {"vec_size", vec_size}, {"voc_size", max_position} };
	VDict kwArdg13 = { {"idx", 2}, {"vec_size", vec_size}, {"voc_size", 2} };

	embed1.appendChild(VModule(m_session, "embed", kwArdg11));
	embed1.appendChild(VModule(m_session, "embed", kwArdg12));
	embed1.appendChild(VModule(m_session, "embed", kwArdg13));

	m_appendChild(embed1);

	VModule encoder(m_session, "sequential", vutils.copy(m_propDict));
	encoder.m_core->m_createTransformerEncoderModel();
	m_appendChild(encoder);

	//m_createTransformerEncoderModel();	// 2 children will be added here

	VModule prune4(m_session, "pruning", { {"name", sent_logit_name } });

	VDict kwArdg41 = { {"axis", 1}, {"index", 0}, {"count", 1},  {"reduce_axis", true} };
	VDict kwArdg42 = { {"in_width", vec_size}, {"out_width", vec_size}, {"actfunc", "tanh"} };
	VDict kwArdg43 = { {"in_width", vec_size}, {"out_width", 2} };

	prune4.appendChild(VModule(m_session, "extract", kwArdg41));
	prune4.appendChild(VModule(m_session, "dense", kwArdg42));
	prune4.appendChild(VModule(m_session, "linear", kwArdg43));

	m_appendChild(prune4);

	VModule prune5(m_session, "pruning", { {"name", word_logit_name} });

	VDict kwArdg51 = { {"in_width", vec_size}, {"out_width", voc_size} };

	prune5.appendChild(VModule(m_session, "linear", kwArdg51));

	m_appendChild(prune5);
}

void VModuleCore::m_createTransformerEncoderModel() {
	int64 stack_depth = vutils.seek_dict(m_propDict, "stack_depth", 12);
	int64 head_cnt = vutils.seek_dict(m_propDict, "head_cnt", 12);
	int64 vec_size = vutils.seek_dict(m_propDict, "vec_size", 768);
	int64 ext_size = vutils.seek_dict(m_propDict, "extend_vector_size", 3072);

	float dropout_inside = vutils.seek_dict(m_propDict, "dropout_inside", 0.1f);
	float dropout_outside = vutils.seek_dict(m_propDict, "dropout_outside", 0.1f);

	VModule att1(m_session, "residual", {});

	VDict kwArdg11 = {};
	VDict kwArdg12 = { {"head_cnt", head_cnt}, {"vec_size", vec_size}, {"dropout", dropout_inside} };
	VDict kwArdg13 = { {"dropout", dropout_outside} };

	att1.appendChild(VModule(m_session, "layernorm", kwArdg11));
	att1.appendChild(VModule(m_session, "mh_attention", kwArdg12));
	att1.appendChild(VModule(m_session, "dropout", kwArdg13));

	m_appendChild(att1);

	VModule post2(m_session, "residual", {});

	VDict kwArdg21 = {};
	VDict kwArdg22 = { {"in_width", vec_size}, {"out_width", vec_size}, {"actfunc", "gelu"} };
	VDict kwArdg23 = { {"in_width", vec_size}, {"out_width", ext_size}, {"actfunc", "gelu"} };
	VDict kwArdg24 = { {"in_width", ext_size}, {"out_width", vec_size}, {"actfunc", "gelu"} };
	VDict kwArdg25 = { {"dropout", dropout_outside} };

	post2.appendChild(VModule(m_session, "layernorm", kwArdg21));
	post2.appendChild(VModule(m_session, "dense", kwArdg22));
	post2.appendChild(VModule(m_session, "dense", kwArdg23));
	post2.appendChild(VModule(m_session, "dense", kwArdg24));
	post2.appendChild(VModule(m_session, "dropout", kwArdg25));

	m_appendChild(post2);

	m_propDict["repeat"] = stack_depth;
}

void VModuleCore::m_createEfficientNetModel(char model_num) {
	if (model_num != '0') {
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}

	//batchnorm 처리를 conv2d 밖으로 끌어낼 것
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);

	int64 target_size = vutils.seek_dict(m_propDict, "target_size", 10);

	VModule SE1(m_session, "squeezeexcitation", {});

	SE1.appendChild(VModule(m_session, "adaptiveavg", {}));
	SE1.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M"}, {"ychn", "#M/4"}, {"batchnorm", false}, {"actfunc", "swish"} }));
	SE1.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M/4"}, {"ychn", "#M"}, {"batchnorm", false}, {"actfunc", "sigmoid"} }));

	VModule MBConv1s(m_session, "sequential", {});

	MBConv1s.appendChild(VModule(m_session, "conv2d", { {"ksize", "#K"}, {"stride", "#S"}, {"group", 0}, {"chn", "#M"}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "swish"} }));
	MBConv1s.appendChild(SE1);
	MBConv1s.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M"}, {"ychn", "#B"}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "none"} }));

	VModule MBConv1r(m_session, "residual", {});

	MBConv1r.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M"}, {"ychn", "#M"}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "swish"} }));
	MBConv1r.appendChild(VModule(m_session, "conv2d", { {"ksize", "#K"}, {"stride", 1}, {"group", 0}, {"chn", "#M"}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "swish"} }));
	MBConv1r.appendChild(SE1);
	MBConv1r.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M"}, {"ychn", "#M"}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "none"} }));
	MBConv1r.appendChild(VModule(m_session, "dropout", { {"drop_ratio", 0.2f} }));

	VModule SE6(m_session, "squeezeexcitation", {});

	SE6.appendChild(VModule(m_session, "adaptiveavg", {}));
	SE6.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M*6"}, {"ychn", "#M/4"}, {"batchnorm", false}, {"actfunc", "swish"} }));
	SE6.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M/4"}, {"ychn", "#M*6"}, {"batchnorm", false}, {"actfunc", "sigmoid"} }));

	VModule MBConv6s(m_session, "sequential", {});

	MBConv6s.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M"}, {"ychn", "#M*6"}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "swish"} }));
	MBConv6s.appendChild(VModule(m_session, "conv2d", { {"ksize", "#K"}, {"stride", "#S"}, {"chn", "#M*6"}, {"group", 0}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "swish"} }));
	MBConv6s.appendChild(SE6);
	MBConv6s.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M*6"}, {"ychn", "#B"}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "none"} }));

	VModule MBConv6r(m_session, "residual", {});

	MBConv6r.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M"}, {"ychn", "#M*6"}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "swish"} }));
	MBConv6r.appendChild(VModule(m_session, "conv2d", { {"ksize", "#K"}, {"stride", 1}, {"chn", "#M*6"}, {"group", 0}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "swish"} }));
	MBConv6r.appendChild(SE6);
	MBConv6r.appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", "#M*6"}, {"ychn", "#M"}, {"use_biias", false}, {"batchnorm", true}, {"actfunc", "none"} }));
	MBConv6r.appendChild(VModule(m_session, "dropout", { {"drop_ratio", 0.2f} }));

	m_session.registMacro("MBConv1s", MBConv1s, {});
	m_session.registMacro("MBConv1r", MBConv1r, {});
	m_session.registMacro("MBConv6s", MBConv6s, {});
	m_session.registMacro("MBConv6r", MBConv6r, {});

	class DropRatio {
	public:
		int m_blocks;
		int m_nth;
		float m_full;
		void set(int blocks, float full) { m_blocks = blocks; m_full = full; m_nth = 0; }
		float get() { return m_full * m_nth++ / m_blocks; }
	};

	DropRatio d;

	d.set(16, 0.2f);

	m_appendChild(VModule(m_session, "conv2d", { {"ksize", 3}, {"stride", 2}, {"xchn", 3}, {"ychn", 32}, {"batchnorm", true}, {"actfunc", "swish"} }));
	m_appendChild(VModule(m_session, "macro", "MBConv1s", { {"K", 3}, {"D", d.get()}, {"M", 32}, {"B", 16}, {"S", 1} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6s", { {"K", 3}, {"D", d.get()}, {"M", 16}, {"B", 24}, {"S", 2} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6r", { {"K", 3}, {"D", d.get()}, {"M", 24} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6s", { {"K", 5}, {"D", d.get()}, {"M", 24}, {"B", 40}, {"S", 2} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6r", { {"K", 5}, {"D", d.get()}, {"M", 40} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6s", { {"K", 3}, {"D", d.get()}, {"M", 40}, {"B", 80}, {"S", 2} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6r", { {"K", 3}, {"D", d.get()}, {"M", 80} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6r", { {"K", 3}, {"D", d.get()}, {"M", 80} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6s", { {"K", 5}, {"D", d.get()}, {"M", 80}, {"B", 112}, {"S", 1} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6r", { {"K", 5}, {"D", d.get()}, {"M", 112} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6r", { {"K", 5}, {"D", d.get()}, {"M", 112} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6s", { {"K", 5}, {"D", d.get()}, {"M", 112}, {"B", 192}, {"S", 2} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6r", { {"K", 5}, {"D", d.get()}, {"M", 192} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6r", { {"K", 5}, {"D", d.get()}, {"M", 192} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6r", { {"K", 5}, {"D", d.get()}, {"M", 192} }));
	m_appendChild(VModule(m_session, "macro", "MBConv6s", { {"K", 3}, {"D", d.get()}, {"M", 192}, {"B", 320}, {"S", 1} }));
	m_appendChild(VModule(m_session, "conv2d", { {"ksize", 1}, {"stride", 1}, {"xchn", 320}, {"ychn", 1280}, {"batchnorm", true}, {"actfunc", "swish"} }));
	m_appendChild(VModule(m_session, "globalavg", {}));
	m_appendChild(VModule(m_session, "batchnorm", { {"out_width", 1280} }));
	m_appendChild(VModule(m_session, "dropout", { {"drop_ratio", 0.2f} }));
	m_appendChild(VModule(m_session, "dense", { {"in_width", 1280}, {"out_width", 512}, {"use_biias", false},  {"batchnorm", true}, {"actfunc", "relu"} }));
	m_appendChild(VModule(m_session, "dense", { {"in_width", 512}, {"out_width", 128}, {"use_biias", false},  {"batchnorm", true}, {"actfunc", "relu"} }));
	m_appendChild(VModule(m_session, "dense", { {"in_width", 128}, {"out_width", target_size} }));

	//VShape shape{ 1, 224, 224, 3 };

	//return m_expandMacro(shape, m_propDict);
}
*/

void VModuleCore::m_appendChild(VModule child) {
	// 기존의 조상-후손 관계 있는지 조사해 있으면 예외처리
	// parent 정보를 두어 검사하면 간단하지만 한 모듈이 여러 군데 이용되지 못한다는 문제가 생긴다.
	// 동일 모듈의 여러 위치 병용은 추후 모델 개발에 유용한 아이디어가 될 수 있으므로 살려둘 필요
	if (child.isDesendent(VModule(this))) VP_THROW2(VERR_RECURSIVE_MODULE_STRUCTURE, m_sName, child.getName());

	if (child.m_core->m_bIncludingMacro) m_bIncludingMacro = true;
	m_children.push_back(child);
	//m_params.push_back(child.params());
}

bool VModule::isDesendent(VModule module) {
	if (m_core == module.m_core) return true;

	for (auto& it: m_core->m_children) {
		if (it.isDesendent(module)) return true;
	}

	return false;
}

VStrList VModuleCore::ms_builtinCustom = {
	"custom"
};

VStrList VModuleCore::ms_builtinLayer = {
	"linear", "addbias", "dense", "flatten", "reshape", "transpose", "concat", 
	"conv2d", "conv2d_transposed", "conv2d_dilated", "max", "avg", "globalavg", "adaptiveavg", "layernorm", "batchnorm", "upsample", "pass",
	"rnn", "lstm", "gru",
	"embed", "dropout", "extract", "mh_attention",
	"activate", "relu", "gelu", "selu", "softmax", "sigmoid", "tanh", "mish", "swish", "leaky",
	"normal_noise", "uniform_noise", "normal_random", "uniform_random", "round", "codeconv", "cosinesim", "selectntop", "selectntoparg",
	"formula"
};

/*
VStrList VModuleCore::ms_builtinModel = {
	"bert", "transformer_encoder",
	"efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7"
};
*/

VStrList VModuleCore::ms_builtinNetwork = {
	"sequential", "parallel", "add", "residual", "pruning", "stack", "squeezeexcitation"
};

bool VModuleCore::ms_inBuiltinCustomNames(string name) {
	for (auto& it : ms_builtinCustom) if (it == name) return true;
	return false;
}

/*
bool VModuleCore::ms_inBuiltinModelNames(string name) {
	for (auto& it : ms_builtinModel) if (it == name) return true;
	return false;
}
*/

bool VModuleCore::ms_inBuiltinLayerNames(string name) {
	for (auto& it : ms_builtinLayer) if (it == name) return true;
	return false;
}

bool VModuleCore::ms_inBuiltinNetworkNames(string name) {
	for (auto& it : ms_builtinNetwork) if (it == name) return true;
	return false;
}

VDict VModuleCore::m_getSerializeInfo(bool bIncludeParam) {
	VDict info;
	
	info["type"] = (int)m_moduleType;
	info["name"] = m_sName;
	info["builtin"] = m_sBuiltin;
	info["props"] = m_propDict;

	info["macro_expanded"] = m_bMacroExpanded;
	info["macro_included"] = m_bIncludingMacro;
	info["macro_args"] = m_macroArgs;

	info["in_shape"] = m_inShape.copy();
	info["out_shape"] = m_outShape.copy();

	info["set_name"] = m_setName;
	info["get_name"] = m_getName;

	info["param_size"] = m_nParamSize;

	VList param_info;

	for (auto& it : m_params) {
		VValue pm_info = m_getParamInfo(it, bIncludeParam);
		param_info.push_back(pm_info);
	}

	info["params"] = param_info;

	VList children_info;

	for (auto& it : m_children) {
		VDict child_info = it.m_core->m_getSerializeInfo(bIncludeParam);
		children_info.push_back(child_info);
	}

	info["children"] = children_info;

	return info;
}

VValue VModuleCore::m_getParamInfo(VValue param, bool bIncludeParam) {
	if (param.is_list()) {
		VList params = param;
		VList params_info;

		for (auto& it : params) {
			VValue pm_info = m_getParamInfo(it, bIncludeParam);
			params_info.push_back(pm_info);
		}

		return params_info;
	}
	else if (param.is_dict()) {
		VDict pm = param;
		VDict pm_info;

		for (auto& it : pm) {
			if (it.second.is_dict()) {
				pm_info[it.first] = m_getPmInfo(it.second, bIncludeParam);
			}
			else {
				pm_info[it.first] = it.second;
			}
		}

		return pm_info;
	}

	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

VDict VModuleCore::m_getPmInfo(VDict pm, bool bIncludeParam) {
	VDict pmset;

	for (auto& it : pm) {
		if (it.first == "pm") {
			if (bIncludeParam) {	// 여기에서 bIncludeParam 따른 별도의 처리가 필요할지?
				pmset[it.first] = (VHTensor)it.second;
			}
			else {
				pmset[it.first] = (VHTensor)it.second;
			}
		}
		else if (it.first == "grad") {
		}
		else {
			if (bIncludeParam) {	// 여기에서 bIncludeParam 따른 별도의 처리가 필요할지?
				pmset[it.first] = (VHTensor)it.second;
			}
			else {
				pmset[it.first] = (VHTensor)it.second;
			}
		}
		/*
		if (it.second.is_tensor()) {
		}
		else {
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
		*/
	}

	return pmset;
}

VValue VModuleCore::m_loadParamInfo(VValue param) {
	if (param.is_list()) {
		VList params = param;
		VList params_info;

		for (auto& it : params) {
			VValue pm_info = m_loadParamInfo(it);
			params_info.push_back(pm_info);
		}

		return params_info;
	}
	else if (param.is_dict()) {
		VDict pm = param;
		VDict pm_info;

		for (auto& it : pm) {
			if (it.second.is_dict()) {
				pm_info[it.first] = m_loadPmInfo(it.second);
			}
			else {
				pm_info[it.first] = it.second;
			}
		}

		return pm_info;
	}

	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

VDict VModuleCore::m_loadPmInfo(VDict pm) {
	VDict pmset;

	for (auto& it : pm) {
		if (it.first == "pm") {
			VTensor param(m_session, (VHTensor)it.second);
			VTensor grad(param, param.shape(), TensorCloneInit::empty);
			pmset["pm"] = param.cloneCore();
			pmset["grad"] = grad.cloneCore();
		}
		else if (it.first == "grad") {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
		else {
			VTensor tensor(m_session, (VHTensor)it.second);
			pmset[it.first] = tensor.cloneCore();
		}
	}

	return pmset;
}

VTensorDict VModuleCore::m_evaluate(VModuleCore* pStarter, VTensorDict xs, bool train, bool noGrad, int nDevice, VTensorDict& sideTerms, VExecTracer tracer) {
	if (m_getName != "") {
		if (xs.find(m_getName) != xs.end()) xs["#"] = xs[m_getName];
		else if (sideTerms.find(m_getName) != sideTerms.end()) xs["#"] = sideTerms[m_getName];
		else VP_THROW2(VERR_GETNAME_NOT_FOUND, m_sBuiltin, m_getName);
	}

	if (m_inShape.size() > 0 && xs.find("#") != xs.end()) {
		VTensor x = xs["#"];
		if (x.isValid() && x.shape().remove_head() != m_inShape.remove_head()) {
			VP_THROW3(VERR_LAYER_EXEC_INSHAPE_MISMATCH, m_sBuiltin, m_inShape.desc(), xs["#"].shape().desc());
		}
	}
	
	VTensorDict ys;
	// 일단 빈 쭉정이 cbInfo 캡슐을 보내고 필요시 본체 생성? pm 변환 포착 칠요!!!
	VCbBackInfo cbInfo(m_session, "");

	m_invokeMatchingCallbacks(pStarter, xs, ys, sideTerms, m_getParameters(), train, noGrad, nDevice, true, cbInfo, tracer);

	if (m_moduleType == VModuleType::layer) {
		VGraph graph = m_getGraph(sideTerms);
		//if (m_sBuiltin == "concat" || m_sBuiltin == "cosinesim") {
		//	graph.setSideTerms(sideTerms);
		//}
		ys = graph.evaluateGraph(xs, train, noGrad, nDevice, cbInfo, tracer);
	}
	else if (m_moduleType == VModuleType::network) {
		// 그래프에 연산 처리를 지시해봐야 네트워크 모듈 특성 및 자식 노드를 이용한 처리 때문에
		// 모듈 정보에 빈번히 접근하거나 아예 다시 모듈에 처리 기능을 만들어 호출해야 한다.
		// 그러느니 모듈에서 직접 처리하기로 한다.
		// 단 오퍼랜드 게산 결과 저장 및 역전파를 위한 대비 등의 이유로 정보 저장이 필요한데
		// 이를 모듈 멤버변수를 이용하여 저장하면 멀티 스레드 실행에서 충돌의 원인이 되고
		// 로컬 변수를 이용하면 역전파 정보를 보존할 수 없다.
		// 따라서 스레드별로 별도 생성되는 그래프 객체를 역전파 정보 저장용 버퍼로 삼으면서 모듈에서 직접 작업을 수행하기로 한다.
		//VTensorDict ys = m_getGraph(sideTerms).evaluate(xs, train, noGrad, nDevice);
		//return ys;
		VGraph graph = m_getGraph(sideTerms);

		if (m_sBuiltin == "sequential") {
			VTensorDict vars = vutils.copy(xs);
			for (auto& it : m_children) {
				VModule child = it;
				vars = child.m_evaluate(pStarter, vars, train, noGrad, nDevice, sideTerms, tracer);
			}
			ys = vars;
		}
		else if (m_sBuiltin == "parallel") {
			VTensorDict xdict;
			int n = 0;
			for (auto& it : m_children) {
				VModule child = it;
				VTensorDict branchs = child.m_evaluate(pStarter, xs, train, noGrad, nDevice, sideTerms, tracer);
				xdict[to_string(n++)] = branchs["#"];
			}
			ys = m_getGraph(sideTerms).evaluateGraph(xdict, train, noGrad, nDevice, cbInfo, tracer);
		}
		else if (m_sBuiltin == "add") {
			VTensorDict xdict;
			int n = 0;
			for (auto& it : m_children) {
				VModule child = it;
				VTensorDict branchs = child.m_evaluate(pStarter, xs, train, noGrad, nDevice, sideTerms, tracer);
				xdict[to_string(n++)] = branchs["#"];
			}
			ys = m_getGraph(sideTerms).evaluateGraph(xdict, train, noGrad, nDevice, cbInfo, tracer);
		}
		else if (m_sBuiltin == "residual") {
			VTensorDict vars = vutils.copy(xs);
			for (auto& it : m_children) {
				VModule child = it;
				vars = child.m_evaluate(pStarter, vars, train, noGrad, nDevice, sideTerms, tracer);
			}
			vars["#residual"] = xs["#"];
			ys = m_getGraph(sideTerms).evaluateGraph(vars, train, noGrad, nDevice, cbInfo, tracer);
		}
		else if (m_sBuiltin == "pruning") {
			VTensorDict vars = vutils.copy(xs);
			for (auto& it : m_children) {
				VModule child = it;
				vars = child.m_evaluate(pStarter, vars, train, noGrad, nDevice, sideTerms, tracer);
			}
			string name = vutils.seek_dict(m_propDict, "name", "");

			if(name != "") {
				xs[name] = vars["#"];
				//VTensor tensor = vars["#"];
				//VTensor shared = tracer.createTensor(tensor, tensor.shape(), TensorCloneInit::share);
				//xs[name] = shared;
			}
			else {
				bool drop = vutils.seek_dict(m_propDict, "drop", false);
				if (!drop) VP_THROW(VERR_PRUNING_WITHOUT_NAME);
				//VTensor tensor = vars["#"];
				//tensor.setNeedGrad(false);
			}
			ys = xs;
		}
		else if (m_sBuiltin == "stack") {
			VTensorDict xdict;
			VTensorDict vars = vutils.copy(xs);
			int n = 0;
			for (auto& it : m_children) {
				VModule child = it;
				VTensorDict branchs = child.m_evaluate(pStarter, vars, train, noGrad, nDevice, sideTerms, tracer);
				xdict[to_string(n++)] = branchs["#"];
			}
			ys = m_getGraph(sideTerms).evaluateGraph(xdict, train, noGrad, nDevice, cbInfo, tracer);
		}
		else if (m_sBuiltin == "squeezeexcitation") {
			VTensorDict vars = vutils.copy(xs);
			for (auto& it : m_children) {
				VModule child = it;
				vars = child.m_evaluate(pStarter, vars, train, noGrad, nDevice, sideTerms, tracer);
			}
			vars["#residual"] = xs["#"];
			ys = m_getGraph(sideTerms).evaluateGraph(vars, train, noGrad, nDevice, cbInfo, tracer);
		}
		else {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
	}
	else if (m_moduleType == VModuleType::custom) {
		VDict result;

		void* pInst;
		void* pAux;

		const VExBuf* pResultBuf;
		VHTensor hResult = 0;

		VCbCustomModuleExec* cbFunc = m_session.getCustomModuleExecCbFunc(&pInst, &pAux);
		VCbFreeReportBuffer* cbFree = m_session.getFreeReportBufferCbFunc(&pInst, &pAux);

		tracer.addMathCall(VMathFunc::__custom_open__, VFuncArgList{});

		VDict xdict = vutils.toDictExternal(xs);
		VDictWrapper wrapper(xdict);

		for (auto& it : xdict) {
			tracer.addMathCall(VMathFunc::__custom_arg__, VFuncArgList{ it.first, (int64)(VHTensor)it.second });
		}

		VObjCore* pModule = (VObjCore*)clone_handle();

		tracer.addMathCall(VMathFunc::__custom_call__, VFuncArgList{ (int64)cbFunc, (int64)cbFree, (int64)pInst, (int64)pAux, (int64)pModule });

		int res_code = cbFunc(pInst, pAux, time(NULL), (VHandle)pModule, wrapper.detach(), &pResultBuf);

		destroy_handle();
		if (res_code != 0) VP_THROW(VERR_INVALID_CUSTOM_MODULE_CALLBACK);

		VDict ydict = VDictWrapper::unwrap(pResultBuf);
		ys = vutils.toTensorDict((VHSession)m_session, ydict);

		for (auto& it : ys) {
			tracer.addTensor(it.second);
		}

		pModule = (VObjCore*)clone_handle();

		res_code = cbFree(pInst, pAux, pResultBuf);
		destroy_handle();
		if (res_code != 0) VP_THROW(VERR_INVALID_FREE_REPORT_BUFFER);
	}
	/*
	else if (m_moduleType == VModuleType::model) {
		VModule child = m_children[0];
		ys = child.m_evaluate(pStarter, xs, train, noGrad, nDevice, sideTerms, tracer);
	}
	*/
	else if (m_moduleType == VModuleType::user_defined) {
		VGraph graph = m_getGraph(sideTerms);
		ys = graph.evaluateGraph(xs, train, noGrad, nDevice, cbInfo, tracer);
	}
	else {
		VP_THROW(VERR_INVALID_MODULE_TYPE);
	}

	for (auto& it : xs) {
		if (it.first != "#" && ys.find(it.first) == ys.end()) {
			ys[it.first] = it.second;
		}
	}

	if (m_setName != "") {
		sideTerms[m_setName] = ys["#"];
	}

	m_invokeMatchingCallbacks(pStarter, xs, ys, sideTerms, m_getParameters(), train, noGrad, nDevice, false, cbInfo, tracer);

	if (m_outShape.size() > 0 && ys["#"].shape().remove_head() != m_outShape.remove_head()) {
		VP_THROW3(VERR_LAYER_EXEC_OUTSHAPE_MISMATCH, m_sBuiltin, m_outShape.desc(), ys["#"].shape().desc());
	}

	return ys;
}

void VModuleCore::m_invokeMatchingCallbacks(
		VModuleCore* pStarter, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params,
		bool train, bool noGrad, int nDevice, bool bPre, VCbBackInfo cbInfo, VExecTracer tracer) {
	if (m_bNeedForwardCallbeck) {
		for (auto& item : m_cbForwardItemMap) {
			int filtered = m_filterCheck(item.second, train, nDevice, bPre);
			if (filtered != 0) continue;
			m_invokeCallback(pStarter, item.second, train, nDevice, bPre, noGrad, xs, ys, sideTerms, params, tracer);
		}
	}

	// 역전파 콜백의 입력 정보는 모듈 실행 진입시에 수집해 두어야 수식그래프 처리 과정에 반영할 수 있다. 따라서 bPre=true일 때 수집을 실행한다.
	// 하지만 처리 후에야 만들어지는 출력 텐서에 대한 처리도 필요하므로 bPre=false일 때에도 추가적인 처리가 필요하다.
	if (m_bNeedBackwardCallbeck && train) {
		for (auto& item : m_cbBackwardItemMap) {
			int filtered = m_filterCheck(item.second, nDevice);
			if (filtered != 0) continue;
			cbInfo.addCbRequestSlot(pStarter, item.second, m_sName, nDevice, bPre, xs, ys, sideTerms, params);
		}
	}

	if (m_session.needCallback()) {
		m_session.invokeMatchingCallbacks(pStarter, m_sName, xs, ys, sideTerms, params, train, noGrad, nDevice, bPre, cbInfo, tracer);
	}
}

int VModuleCore::m_filterCheck(VCbItem item, bool train, int nDevice, bool bPre) {
	VDict filters = item.getFilters();

	for (auto& filter : filters) {
		VList noms = filter.second;

		if (filter.first == "mode") {
			if (train && !noms.find_string("train")) return 3;
			if (!train && !noms.find_string("test")) return 4;
		}
		else if (filter.first == "device") {
			if (noms.find(nDevice) == noms.end()) return 7;
		}
		else if (filter.first == "phase") {
			if (bPre && !noms.find_string("pre")) return 8;
			if (!bPre && !noms.find_string("post")) return 9;
		}
	}

	return 0;
}

int VModuleCore::m_filterCheck(VCbItem item, int nDevice) {
	VDict filters = item.getFilters();

	for (auto& filter : filters) {
		VList noms = filter.second;

		if (filter.first == "device") {
			if (noms.find(nDevice) == noms.end()) return 7;
		}
	}

	return 0;
}

void VModuleCore::m_invokeCallback(
		VModuleCore* pStarter, VCbItem item, bool train, int nDevice, bool bPre,
		bool noGrad, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params, VExecTracer tracer) {
	//string name = (m_propDict.find("name") != m_propDict.end()) ? (string)m_propDict["name"] : "#" + to_string(getNth());

	m_session.invokeCallback(pStarter, item, m_sName, train, nDevice, bPre, noGrad, xs, ys, sideTerms, params, tracer);

	/*
	VDict instInfo = item.m_instInfo;

	instInfo["#data_idx"] = pStarter->getDataIdx(nDevice);

	VDict statusInfo;

	statusInfo["name"] = (m_propDict.find("name") != m_propDict.end()) ? (string)m_propDict["name"] : "#" + to_string(getNth());
	statusInfo["mode"] = train ? "train" : "test";
	statusInfo["device"] = nDevice;
	statusInfo["phase"] = bPre ? "pre" : "post";
	statusInfo["no_grad"] = noGrad;

	VList names = { "input", "output", "sideterm", "param" };

	VTensorDict tensors[4];

	tensors[0] = xs;
	tensors[1] = ys;
	tensors[2] = sideTerms;
	tensors[3] = params.getWeights();

	VDict tensorDict = vutils.toDictExternal(tensors, names);

	tracer.addInvokeForwardCallback(item.m_cbFunc, item.m_cbClose, instInfo, statusInfo, tensorDict);

	extern VDict V_invokeModuleForwardCallback(VHSession hSession, VCbForwardModule * pCbFunc, VCbClose * pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict);

	// 콜백 반값 result는 아직 특별한 용도가 없지만 차후 확장에 대비해 전달받을 수 있게 한다.
	VDict result = V_invokeModuleForwardCallback(m_session, item.m_cbFunc, item.m_cbClose, instInfo, statusInfo, tensorDict);
	*/
}

void VModuleCore::m_splitDataIdx(int nDivisions) {
	if (m_dataIdx.size() <= 0 || nDivisions <= 1) return;

	bool bCreate = m_dataIdxes.size() == 0;

	int64 dataCount = m_dataIdx.size();
	int64 pieceSize = dataCount / nDivisions;

	for (int64 n = 0, k = 0; n < nDivisions; n++) {
		if (bCreate) {
			VList partIdx;
			for (int64 m = 0; m < pieceSize; m++) {
				partIdx.push_back(m_dataIdx[k++]);
			}
			m_dataIdxes.push_back(partIdx);
		}
		else {
			VList partIdx = m_dataIdxes[n];
			for (int64 m = 0; m < pieceSize; m++) {
				partIdx[m] = m_dataIdx[k++];
			}
		}
	}
}

VParameters VModuleCore::m_getParameters() {
	if (!m_parameters.isValid()) {
		m_parameters = VParameters(m_session, m_params, m_sDevice);
	}

	return m_parameters;
}


VGraph VModuleCore::m_getGraph(VTensorDict& sideTerms) {
	int nDevice = m_session.device_man().getCurDevice();

	if (nDevice < 0) return m_graphTemplate;

	if (m_graphMap.find(nDevice) == m_graphMap.end()) {
		VGraph graph(GraphInit::deep_copy, m_session.device_man(), m_graphTemplate);
		m_graphMap[nDevice] = graph;
	}

	VGraph graph = m_graphMap[nDevice];
	graph.setSideTerms(sideTerms);
	return graph;
}
