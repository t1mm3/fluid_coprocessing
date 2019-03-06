#include "query.hpp"

#include <vector>
#include <thread>

struct Scheduler {
private:
	std::vector<std::thread> workers;

	std::atomic<bool> done;

	template<bool GPU>
	void worker() {
		while (!done) {
			
		}
	}

	void NO_INLINE cpu() { return worker<false>(); }
	void NO_INLINE gpu() { return worker<true>(); }
public:
	~Scheduler() {
		for (auto& w : workers) {
			w.join();
		}
	}

	Scheduler() {
		std::thread t(&Scheduler::cpu, this);
	}
};