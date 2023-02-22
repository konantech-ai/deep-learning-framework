#include "../local_objects/vtensordata.h"
#include "../local_objects/vtensordata_core.h"
#include "../support/vmath.h"

VTensorData::VTensorData() {
	m_core = NULL;
}

VTensorData::VTensorData(VSession session, string sBuiltin, VDict kwArgs) {
	m_core = new VTensorDataCore(session, sBuiltin, kwArgs);
}

VTensorData::VTensorData(const VTensorData& src) {
	m_core = src.m_core->clone();
}

VTensorData::VTensorData(VTensorDataCore* core) {
	m_core = core->clone();
}

VTensorData::~VTensorData() {
	m_core->destroy();
}

VTensorData& VTensorData::operator =(const VTensorData& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VTensorDataCore* VTensorData::getClone() {
	return (VTensorDataCore*)m_core->clone_core();
}

VTensorDataCore* VTensorData::getCore() {
	return m_core;
}

void VTensorData::destroyCore() {
	if (m_core->getRefCnt() > 1) m_core->destroy();
	else {
		m_core->destroy();
		m_core = NULL;
	}
}

VSession VTensorData::session() const {
	return m_core->m_session;
}

bool VTensorData::isValid() {
	return m_core != NULL;
}

int VTensorData::getRefCnt() {
	return m_core->getRefCnt();
}

int VTensorData::getNth() {
	return m_core->getNth();
}

VTensorDataCore::VTensorDataCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::TensorData) {
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_session = session,
		m_setup();
}

VTensorData::VTensorData(VSession session, int64 byteSize, int device) {
	m_core = new VTensorDataCore(session);

	m_core->m_nDevice = device;
	m_core->m_byteSize = byteSize;
	m_core->m_ptr = VMath::mem_alloc(device, byteSize);
}

int64 VTensorData::byte_size() {
	return m_core ? m_core->m_byteSize : 0;
}

void* VTensorData::void_ptr() const {
	return m_core ? m_core->m_ptr : 0;
}

int* VTensorData::int_ptr() const {
	return m_core ? (int*)m_core->m_ptr : 0;
}

int64* VTensorData::int64_ptr() const {
	return m_core ? (int64*)m_core->m_ptr : 0;
}

float* VTensorData::float_ptr() const {
	return m_core ? (float*)m_core->m_ptr : 0;
}

unsigned char* VTensorData::uchar_ptr() const {
	return m_core ? (unsigned char*)m_core->m_ptr : 0;
}

int VTensorData::device() {
	return m_core ? m_core->m_nDevice : -1;
}

void VTensorData::uploadData(void* pData, int64 nByteSize) {
	if (nByteSize != m_core->m_byteSize) VP_THROW(VERR_NOT_IMPLEMENTED_YET);

	if (m_core->m_nDevice < 0) {
		VMath::memcpy_host_to_host(m_core->m_ptr, pData, m_core->m_byteSize);
	}
	else {
		VMath::memcpy_host_to_device(m_core->m_ptr, pData, m_core->m_byteSize);
	}
}

void VTensorData::downloadData(void* pData, int64 nByteSize) {
	if (nByteSize != m_core->m_byteSize) VP_THROW(VERR_NOT_IMPLEMENTED_YET);

	if (m_core->m_nDevice < 0) {
		VMath::memcpy_host_to_host(pData, m_core->m_ptr, m_core->m_byteSize);
	}
	else {
		VMath::memcpy_device_to_host(pData, m_core->m_ptr, m_core->m_byteSize);
	}
}

void VTensorData::setZero() {
	m_core->m_memset(0);
}

/*
void VTensorData::copyFrom(VTensorData src) {
	if (m_core->m_nDevice < 0) {
		if (src.m_core->m_nDevice < 0) {
			VMath::memcpy_host_to_host(m_core->m_ptr, src.m_core->m_ptr, m_core->m_byteSize);
		}
		else {
			VMath::memcpy_device_to_host(m_core->m_ptr, src.m_core->m_ptr, m_core->m_byteSize);
		}
	}
	else {
		if (src.m_core->m_nDevice < 0) {
			VMath::memcpy_host_to_device(m_core->m_ptr, src.m_core->m_ptr, m_core->m_byteSize);
		}
		else {
			VMath::memcpy_device_to_device(m_core->m_ptr, src.m_core->m_ptr, m_core->m_byteSize);
		}
	}
}
*/

void VTensorData::memset(int value) {
	m_core->m_memset(value);
}

void VTensorData::fill_float(float value) {
	m_core->m_fill_float(value);
}

void VTensorData::init_random_normal(float mean, float init_arg, bool adaptive) {
	m_core->m_init_random_normal(mean, init_arg, adaptive);
}

void VTensorData::init_random_uniform(float mean, float init_arg) {
	m_core->m_init_random_uniform(mean, init_arg);
}

void VTensorDataCore::m_setup() {
	m_nDevice = -1;
	m_ptr = NULL;
	m_byteSize = 0;
}

VTensorDataCore::~VTensorDataCore() {
	m_freeData();
}

void VTensorDataCore::m_freeData() {
	VMath::mem_free(m_nDevice, m_ptr);
}

void VTensorDataCore::m_memset(int value) {
	int* ptr = (int*)m_ptr;
	int64 size = m_byteSize / sizeof(int);
	VMath::fill_int(m_nDevice, ptr, size, value);
}

void VTensorDataCore::m_fill_float(float value) {
	float* ptr = (float*)m_ptr;
	int64 size = m_byteSize / sizeof(float);
	VMath::fill_float(m_nDevice, ptr, size, value);
}

void VTensorDataCore::m_init_random_normal(float mean, float init_arg, bool adaptive) {
	float* ptr = (float*)m_ptr;
	int64 size = m_byteSize / sizeof(float);

	VMath::init_random_normal(m_nDevice, ptr, size, mean, init_arg, adaptive);
}

void VTensorDataCore::m_init_random_uniform(float mean, float init_arg) {
	float* ptr = (float*)m_ptr;
	int64 size = m_byteSize / sizeof(float);

	VMath::init_random_uniform(m_nDevice, ptr, size, mean, init_arg);
}
