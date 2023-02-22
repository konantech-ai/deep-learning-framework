#include "../include/vapi.h"
#include "../utils/vexception.h"
#include "../support/vmath.h"

#if defined(V_DEBUG_OBJ_LEAK)
void dumpObjectUsage(string title) {
	static mutex dump_mutex;

	dump_mutex.lock();
	printf("[Dump Object Usage: %s]\n", title.c_str());
	VObjCore::DumpUsage();
	dump_mutex.unlock();
}

void VObjCore::DumpUsage() {
	ms_ref_mutex.lock();
	printf("%d objects were created, %d objects are still alive.\n", ms_nNextId, ms_nInstCount);
	int nobj = 0;
	printf("\nObjects: ");
	for (auto& it : ms_instAlive) {
		printf(" #%d(%d, %d)", it.second->getNth(), it.second->getType(), it.second->getRefCnt());
		if (nobj >= 49) break;
		if (nobj++ % 10 == 9) printf("\n");
	}
	printf("\n...\n         ");

	nobj = 0;
	for (map<int, VObjCore*>::reverse_iterator it = ms_instAlive.rbegin(); it != ms_instAlive.rend(); it++) {
		printf(" #%d(%d, %d)", it->second->getNth(), it->second->getType(), it->second->getRefCnt());
		if (nobj >= 49) break;
		if (nobj++ % 10 == 9) printf("\n");
	}
	printf("\n\n");

	VMath::DumpUsage();
	ms_ref_mutex.unlock();
}
#else
void dumpObjectUsage(string title) {
	printf("dumpObjectUsage() 함수는 디버그 버전에서만 사용 가능한 기능입니다.\n");
}
#endif
