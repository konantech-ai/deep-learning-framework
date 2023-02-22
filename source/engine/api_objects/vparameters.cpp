#include "../api_objects/vparameters.h"
#include "../api_objects/vparameters_core.h"
#include "../api_objects/vtensor.h"
#include "../local_objects/vdevicemanager.h"
#include "../support/vmath.h"

int VParametersCore::ms_nCheckCode = 25995644;

//=========== API Object Common Part Start =======================

VParameters::VParameters() {
	m_core = NULL;
}

VParameters::VParameters(const VParameters& src) {
	m_core = src.m_core->clone();
}

VParameters::VParameters(VParametersCore* core) {
	m_core = core->clone();
}

VParameters::VParameters(VSession session, string sBuiltin, VDict kwArgs) {
	m_core = new VParametersCore(session, sBuiltin, kwArgs);
}

VParameters::VParameters(VSession session, VHParameters handle) {
	m_core = NULL;
	VParametersCore* core = (VParametersCore*)handle;
	if (core == NULL) VP_THROW1(VERR_INVALID_CORE, "Parameters");
	if (core->m_nCheckCode != VParametersCore::ms_nCheckCode) VP_THROW1(VERR_NOT_EQUAL_CORE_CHECKCODE, "Parameters");
	if (core->m_session != session) VP_THROW1(VERR_NOT_EQUAL_CORE_SESSION, "Parameters");
	m_core = (VParametersCore*)core->clone_core();
}

VParameters::~VParameters() { m_core->destroy(); }

VParameters& VParameters::operator =(const VParameters& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VHParameters VParameters::cloneCore() {
	return (VHParameters)m_core->clone();
}

VHParameters VParameters::cloneHandle() {
	return (VHParameters)m_core->clone_handle();
}

VParametersCore* VParameters::getClone() {
	return (VParametersCore*)m_core->clone_core();
}

VParametersCore* VParameters::getCore() {
	return m_core;
}

bool VParameters::isValid() {
	return m_core != NULL;
}
void VParameters::closeHandle() {
	if (this) m_core->destroy_handle();
}

VSession VParameters::session() {
	return m_core->m_session;
}

int VParameters::getRefCnt() {
	return m_core->getRefCnt();
}

int VParameters::getNth() {
	return m_core->getNth();
}

void VParameters::incRefCount() {
	m_core->incRefCnt();
}

VParametersCore::VParametersCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::Parameters) {
	m_nCheckCode = ms_nCheckCode;
	m_session = session;
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_onCreate();
}

VParametersCore::~VParametersCore() {
	m_onDelete();
	m_nCheckCode = 0;
}

//=========== API Object Common Part End =======================

VParameters::VParameters(VSession session, VList params, string sDevice) {
	m_core = new VParametersCore(session, "param", {});
	m_core->m_params = params;
	m_core->m_sDevice = sDevice;
}

string VParameters::getDevice() {
	return m_core->m_sDevice;
}

void VParameters::getWeights(VList& terms, VTensorDict& weights, bool bGrad) {
	VDict nums;
	string kind = bGrad ? "grad" : "pm";
	m_lookupTensors(terms, weights, nums, m_core->m_params, kind, "#layer");
}

void VParameters::zero_grad() {
	VExecTracer empty_tracer;

	//int nDevice = (session().device_man().getCurDevice() < 0) ? -1 : 0;
	int nDevice = (m_core->m_sDevice == "cpu") ? -1 : 0;
	int nOldDevice = session().device_man().setCurDevice(nDevice, empty_tracer);
	m_core->m_zero_grad(m_core->m_params, nDevice);
	session().device_man().setCurDevice(nOldDevice, empty_tracer);
}

void VParameters::init_weights() {
	VExecTracer empty_tracer;

	//int nDevice = (session().device_man().getCurDevice() < 0) ? -1 : 0;
	int nDevice = (m_core->m_sDevice == "cpu") ? -1 : 0;
	int nOldDevice = session().device_man().setCurDevice(nDevice, empty_tracer);
	m_core->m_init_weights(m_core->m_params, nDevice);
	session().device_man().setCurDevice(nOldDevice, empty_tracer);
}

VList VParameters::getParams() {
	return m_core->m_params;
}

void VParameters::load_param(VDict pmset, string name, FILE* fid) {
	VTensor pm((VTensorCore*)(VObjCore*)pmset["pm"]);

	if (pm.hasNoData()) {
		return;
	}

	int64 size = pm.shape().total_size();
	float* ppm = pm.float_ptr();

	if (name == "rescale" || name == "bias" || name == "mavg" || name == "mvar") {
		if (fread(ppm, sizeof(float), size, fid) != size) VP_THROW(VERR_SIZE_PARAMETER);
	}
	else if (name == "kernel") {
		VShape shape = pm.shape();
		VShape tshape = { shape[3], shape[2], shape[0], shape[1] };
		VTensor temp(pm.session(), tshape, VDataType::float32, -1);
		float* ptm = temp.float_ptr();
		if (fread(ptm, sizeof(float), size, fid) != size) VP_THROW(VERR_SIZE_PARAMETER);
		pm.transpose_on(temp, VList{ 2, 3 , 1, 0 }, VExecTracer());
	}
	else {
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}
}

void VParameters::m_lookupTensors(VList& terms, VTensorDict& tensors, VDict& nums, VList params, string kind, string def_name) {
	int nth = 0;

	for (auto& it : params) {
		if (it.is_list()) m_lookupTensors(terms, tensors, nums, (VList)it, kind, def_name + "." + to_string(nth++));
		else if (it.is_dict()) m_lookupTensors(terms, tensors, nums, (VDict)it, kind, def_name);
		else {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
	}
}

void VParameters::m_lookupTensors(VList& terms, VTensorDict& tensors, VDict& nums, VDict params, string kind, string def_name) {
	string name = params["module_name"];
	if (name == "") name = def_name;
	int nth = (nums.find(name) == nums.end()) ? 0 : (int)nums[name];

	nums[name] = nth + 1;

	for (auto& it : params) {
		if (it.second.is_dict()) {
			VDict pmset = it.second;
			if (pmset.find(kind) == pmset.end()) continue;
			string pm_name = name + "." + it.first;
			VTensor tensor(session(), (VHTensor)pmset[kind]);
			
			VDict pm_term;
			pm_term["name"] = name;
			pm_term["type"] = it.first;
			pm_term["key"] = pm_name;

			terms.push_back(pm_term);
			tensors[pm_name] = tensor;
		}
	}
}

/*
void VParameters::m_lookupTensors(VTensorDict& tensors, VDict& nums, VDict params, string kind, string def_name) {
	string name = params["module_name"];
	if (name == "") name = def_name;
	int nth = (nums.find(name) == nums.end()) ? 0 : (int)nums[name];
	
	nums[name] = nth + 1;

	for (auto& it : params) {
		if (it.second.is_dict()) {
			VDict pmset = it.second;
			if (pmset.find(kind) == pmset.end()) continue;
			string pm_name = name + "." + it.first;
			VTensor tensor(session(), (VHTensor)pmset[kind]);
			tensors[pm_name] = tensor;
		}
	}
}
*/

void VParametersCore::m_onCreate() {}
void VParametersCore::m_onDelete() {}

void VParametersCore::m_zero_grad(VList params, int nDevice) {
	for (auto& it : params) {
		if (it.is_list()) m_zero_grad((VList)it, nDevice);
		else if (it.is_dict()) {
			VDict param = it;
			for (auto& it2 : param) {
				if (it2.second.is_dict()) {
					VDict pmset = it2.second;
					VTensor grad(m_session, (VHTensor)pmset["grad"]);
					if (grad.hasNoData()) grad.allocData(nDevice);
					grad.setZero(VExecTracer());	// 텐서 할당이 관련되지 않으므로 간단히 VExecTracer 사용 않아도 무방해 보임
				}
			}
		}
		else {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
	}
}

void VParametersCore::m_init_weights(VList params, int nDevice) {
	for (auto& it : params) {
		if (it.is_list()) m_init_weights((VList)it, nDevice);
		else if (it.is_dict()) {
			VDict param = it;
			for (auto& it2 : param) {
				if (it2.second.is_dict()) {
					VDict pmset = it2.second;
					VTensor pm(m_session, (VHTensor)pmset["pm"]);

					pm.initParam();
				}
			}
		}
		else {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
	}
}
