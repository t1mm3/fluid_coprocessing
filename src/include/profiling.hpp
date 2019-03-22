/* Copyright (c) 2019 by Tim Gubner, CWI
 * Licensed under GPLv3
 */

#pragma once

struct Profiling {
	struct Time {
		uint64_t cycles = 0;

		static Time diff(const Time& stop, const Time& start) {
			return Time {stop.cycles - start.cycles};
		}

		void aggregate(Time t) {
			cycles += t.cycles;
		}

		void atomic_aggregate(Time t) {
			__sync_add_and_fetch(&cycles, t.cycles);
		}

		void reset() {
			cycles = 0;
		}
	};

	static Time start(bool gpu = false) {
		return Time { rdtsc() };
	}

	static Time stop(bool gpu = false) {
		return start();
	}

	struct Scope {
		Time& m_aggr;
		Time m_start;
		bool m_gpu;

		Scope(Time& dest, bool gpu = false) : m_aggr(dest), m_gpu(gpu) {
			m_start = start(gpu);
		}

		~Scope() {
			m_aggr.aggregate(Time::diff(stop(m_gpu), m_start));
		}
	};
};
