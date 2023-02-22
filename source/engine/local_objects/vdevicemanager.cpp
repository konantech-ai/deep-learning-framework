#include "../local_objects/vdevicemanager.h"
#include "../local_objects/vdevicemanager_core.h"
#include "../local_objects/vexectracer.h"
#include "../local_objects/vexectracer_core.h"
#include "../support/vmath.h"

VDeviceManager::VDeviceManager() {
    m_core = NULL;
}

VDeviceManager::VDeviceManager(string sBuiltin, VDict kwArgs) {
    m_core = new VDeviceManagerCore(sBuiltin, kwArgs);
}

VDeviceManager::VDeviceManager(const VDeviceManager& src) {
    m_core = src.m_core->clone();
}

VDeviceManager::VDeviceManager(VDeviceManagerCore* core) {
    m_core = core->clone();
}

VDeviceManager::~VDeviceManager() {
    m_core->destroy();
}

VDeviceManager& VDeviceManager::operator =(const VDeviceManager& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

VDeviceManagerCore* VDeviceManager::getClone() {
    return (VDeviceManagerCore*)m_core->clone_core();
}

bool VDeviceManager::isValid() {
    return m_core != NULL;
}

int VDeviceManager::getRefCnt() {
    return m_core->getRefCnt();
}

int VDeviceManager::getNth() {
    return m_core->getNth();
}

VDeviceManagerCore::VDeviceManagerCore(string sBuiltin, VDict kwArgs) : VObjCore(VObjType::DeviceManager) {
    m_sBuiltin = vutils.tolower(sBuiltin);
    m_propDict = kwArgs;
    m_setup();
}

//--------------------------------------------------------------------------------------------------
//
int VDeviceManager::getDeviceCount() {
    int nAvailDevices;
    VMath::cudaCheck(cudaGetDeviceCount(&nAvailDevices), "cudaGetDeviceCount", __FILE__, __LINE__);
    return nAvailDevices;
}

void VDeviceManager::setUsingCudaFlag(int nModuleId, int flag) {
    m_core->m_moduleDeviceFlags[nModuleId] = flag;
}

int VDeviceManager::getUsingDeviceCount(int nModuleId) {
    int flag = m_core->m_moduleDeviceFlags[nModuleId];
    int using_dev_count = 0;
    int dev_count = getDeviceCount();

    for (int n = 0; n < dev_count; n++) {
        if (flag & (0x01 << n)) {
            using_dev_count++;
        }
    }
    
    return using_dev_count;
}

int VDeviceManager::getNthUsingDevice(int nModuleId, int nth) {
    int flag = m_core->m_moduleDeviceFlags[nModuleId];
    int dev_count = getDeviceCount();

    for (int n = 0; n < dev_count; n++) {
        if (flag & (0x01 << n)) {
            if (nth-- <= 0) {
                return n;
            }
        }
    }

    VP_THROW(VERR_UNDEFINED);

    return -1;
}

int VDeviceManager::getCurDevice() {
    std::thread::id this_id = std::this_thread::get_id();

    if (m_core->m_curDevice.find(this_id) == m_core->m_curDevice.end()) {
        return -1; // VP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return m_core->m_curDevice[this_id];
}

int VDeviceManager::setCurDevice(int nDevice, VExecTracer tracer) {
    int nOldDevice = getCurDevice();

    std::thread::id this_id = std::this_thread::get_id();

    m_core->m_curDevice[this_id] = nDevice;

    if (nDevice >= 0) {
        VMath::cudaCheck(cudaSetDevice(nDevice), "cudaSetDevice", __FILE__, __LINE__);
    }
    tracer.addMathCall(VMathFunc::__set_curr_device__, { nDevice });
    return nOldDevice;
}

int64 VDeviceManager::getMaxBatchSize(int64 nNeedSizePerbatch) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free / nNeedSizePerbatch;	// 실제로 fragmentation 때문에 할당이 불가능할 수도 있음
}

void VDeviceManagerCore::m_setup() {
    m_moduleDeviceFlags.clear();
}
