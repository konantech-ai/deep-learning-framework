#include "../local_objects/vhypermanager.h"
#include "../local_objects/vhypermanager_core.h"
#include "../support/vmath.h"

VHyperManager::VHyperManager() {
	m_core = NULL;
}

VHyperManager::VHyperManager(string sBuiltin, VDict kwArgs) {
	m_core = new VHyperManagerCore(sBuiltin, kwArgs);
}

VHyperManager::VHyperManager(const VHyperManager& src) {
	m_core = src.m_core->clone();
}

VHyperManager::VHyperManager(VHyperManagerCore* core) {
	m_core = core->clone();
}

VHyperManager::~VHyperManager() {
	m_core->destroy();
}

VHyperManager& VHyperManager::operator =(const VHyperManager& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VHyperManagerCore* VHyperManager::getClone() {
	return (VHyperManagerCore*)m_core->clone_core();
}

bool VHyperManager::isValid() {
	return m_core != NULL;
}

int VHyperManager::getRefCnt() {
	return m_core->getRefCnt();
}

int VHyperManager::getNth() {
	return m_core->getNth();
}

VHyperManagerCore::VHyperManagerCore(string sBuiltin, VDict kwArgs) : VObjCore(VObjType::HyperManager) {
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_setup();
}

//--------------------------------------------------------------------------------------------------
//
int VHyperManager::regist() {
	if (m_core->m_nUsingCount >= m_core->m_nAllocSize) {
		m_core->m_allocAdditionalPage();
	}

	return m_core->m_nUsingCount++;
}

int VHyperManager::registValue(float value) {
	int key = regist();
	m_core->m_setValue(key, value);
	return key;
}

void VHyperManager::set(int key, float value) {
	m_core->m_setValue(key, value);
}

float VHyperManager::get(int key) {
	return m_core->m_pHyperBase[key];
}

float* VHyperManager::fetch(int device, int key) {
	if (device < -1) VP_THROW(VERR_UNDEFINED);

	int page = key / PAGE_SIZE;
	int offset = key % PAGE_SIZE;

	m_core->m_dirtyCheck(device, page);

	return m_core->m_hyperMaps[device][page] + offset;
}

//--------------------------------------------------------------------------------------------------

void VHyperManagerCore::m_setup() {
	m_nAllocSize = PAGE_SIZE;
	m_nUsingCount = 0;

	m_pHyperBase = (float*)malloc(m_nAllocSize * sizeof(float));

	if (m_pHyperBase == NULL) VP_THROW(VERR_HOSTMEM_ALLOC_FAILURE);

	memset(m_pHyperBase, 0, m_nAllocSize * sizeof(float));
}

VHyperManagerCore::~VHyperManagerCore() {
	free(m_pHyperBase);

	for (auto& maps : m_hyperMaps) {
		for (auto& map : maps.second) {
			VMath::mem_free(maps.first, map);
		}
	}
}

void VHyperManagerCore::m_allocAdditionalPage() {
	int nExtendSize = m_nAllocSize + PAGE_SIZE;
	float* pExtendBase = (float*)malloc(nExtendSize * sizeof(float));

	if (pExtendBase == NULL) VP_THROW(VERR_HOSTMEM_ALLOC_FAILURE);

	memset(pExtendBase, 0, nExtendSize * sizeof(float));
	memcpy(pExtendBase, m_pHyperBase, m_nAllocSize * sizeof(float));
	
	free(m_pHyperBase);
	
	m_pHyperBase = pExtendBase;
	m_nAllocSize = nExtendSize;
}

void VHyperManagerCore::m_setValue(int key, float value) {
	m_pHyperBase[key] = value;

	int page = key / PAGE_SIZE;

	for (auto& flags : m_dirtyMaps) {
		flags.second[page] = true;
	}
}

void VHyperManagerCore::m_dirtyCheck(int device, int page) {
	if (m_dirtyMaps.find(device) == m_dirtyMaps.end()) {
		m_dirtyMaps[device] = vector<bool>();
		m_hyperMaps[device] = vector<float*>();
	}

	while (m_dirtyMaps[device].size() <= page) {
		float* pMap = (float*)VMath::mem_alloc(device, PAGE_SIZE * sizeof(float));
		m_dirtyMaps[device].push_back(true);
		m_hyperMaps[device].push_back(pMap);
	}

	if (m_dirtyMaps[device][page]) {
		VMath::memcpy_from_host(device, m_hyperMaps[device][page], m_pHyperBase + PAGE_SIZE * page, PAGE_SIZE * sizeof(float));
		m_dirtyMaps[device][page] = false;
	}
}
