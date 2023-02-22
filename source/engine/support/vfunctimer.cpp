/*
#include "../support/vfunctimer.h"

VDict VFunctionTimer::ms_records;
mutex VFunctionTimer::ms_mu_ftimer;

VFunctionTimer::VFunctionTimer(string name) {
	m_name = name;
	m_start_time = high_resolution_clock::now();
}

VFunctionTimer::~VFunctionTimer() {
	high_resolution_clock::time_point  end_time = high_resolution_clock::now();

	duration<double, std::ratio<1, 1000000000>> time_span = duration_cast<duration<double, std::ratio<1, 1000000000>>>(end_time - m_start_time);

	int64 mtime = (int64)time_span.count();

	VDict records;

	ms_mu_ftimer.lock();

	if (ms_records.find(m_name) != ms_records.end()) {
		records = ms_records[m_name];
	}
	else {
		records["num"] = 0;
		records["sum"] = (int64)0;
		records["sqsum"] = (int64)0;
	}

	records["num"] = (int)records["num"] + 1;
	records["sum"] = (int64)records["sum"] + mtime;
	records["sqsum"] = (int64)records["sqsum"] + mtime * mtime;

	ms_records[m_name] = records;
	ms_mu_ftimer.unlock();
}

void VFunctionTimer::init() {
	ms_mu_ftimer.lock();
	ms_records.clear();
	ms_mu_ftimer.unlock();
}

void VFunctionTimer::dump() {
	printf("*** Function execution times ***\n");
	printf("%-30s %15s %15s %15s %15s\n", "name", "num", "sum", "avg", "std");
	int total_num = 0;
	int64 total_sum = 0;
	int64 total_sqsum = 0;
	VList names;
	VList avgs;
	VList ranks;
	ms_mu_ftimer.lock();

	int64 cnt = ms_records.size();

	for (auto& it : ms_records) {
		string name = it.first;
		VDict records = it.second;
		int num = records["num"];
		int64 sum = records["sum"];
		int64 sqsum = records["sqsum"];
		float avg = (float)sum / (float)num;
		names.push_back(name);
		avgs.push_back(avg);
		ranks.push_back(0);
		total_num += num;
		total_sum += sum;
		total_sqsum += sqsum;
	}

	for (int64 n = 0; n < cnt; n++) {
		for (int64 m = n + 1; m < cnt; m++) {
			if ((float)avgs[n] < (float)avgs[m]) ranks[n] = (int64)ranks[n] + 1;
			else ranks[m] = (int64)ranks[m] + 1;
		}
	}

	for (int64 n = 0; n < cnt; n++) {
		for (int64 m = 0; m < cnt; m++) {
			if ((int64)ranks[m] != n) continue;

			VDict records = ms_records[names[m]];
			int num = records["num"];
			int64 sum = records["sum"];
			int64 sqsum = records["sqsum"];
			float avg = avgs[m];
			float std = (float) ::sqrt((float)sqsum / (float)num - avg * avg);
			printf("%-30s %15d %15lld %15.3f %15.3f\n", ((string)names[m]).c_str(), num, sum, avg, std);
		}
	}
	ms_mu_ftimer.unlock();
	float avg = (float)total_sum / (float)total_num;
	float std = (float) ::sqrt((float)total_sqsum / (float)total_num - avg * avg);
	printf("%-30s %15d %15lld %15.3f %15.3f\n", "TOTAL", total_num, total_sum, avg, std);
}
*/