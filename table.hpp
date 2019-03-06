#pragma once

#include <stdint.h>
#include <atomic>
#include <cassert>

struct Table {
private:
	int64_t capacity;
	std::atomic<int64_t> start; //! Current start offset
	std::atomic<int64_t> done; //!< Current finished offset (after morsel completion)

	void *base;

public:
	int64_t size() const { return capacity; }

	Table(int64_t cap) {
		start = 0;
		done = 0;
		capacity = cap;
	}

	void reset() {
		done = 0;
		start = 0;
	}

private:
	/* "Consumes" a range of the table
		\param ostart Offset of range
		\param onum Number of tuples in range
		\param num Preferred size of range
		\returns True, if successful (and valid range); False, otherwise
	*/
	bool get_range(int64_t& onum, int64_t& ostart, int64_t num) {
		if (start > capacity) {
			return false;
		}

		ostart = std::atomic_fetch_add(&start, num);
		if (ostart >= capacity) {
			return false;
		}

		const int64_t todo = capacity - ostart;
		assert(todo > 0);

		onum = std::min(todo, num);
		return onum > 0;
	}

public:
	template<typename T, typename S>
	void chunk(T&& morsel, S&& finished, int64_t morsel_size = 16*1024) {
		while (1) {
			int64_t num = 0;
			int64_t offset = 0;
			bool has_range = get_range(num, offset, morsel_size);

			if (!has_range) {
				break;
			}

			assert(num <= morsel_size);

			morsel(base, offset, num);

			done += num;
			if (done >= capacity) {
				finished();
			}
		}
	}
};
