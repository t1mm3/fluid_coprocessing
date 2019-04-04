#ifndef H_TIMELINE
#define H_TIMELINE

#include <iostream>
#include <vector>
#include <stdint.h>
#include <atomic>

template<typename Event>
struct Timeline {
protected:
	static constexpr size_t kEventBufSize = 1024*64;
	
	struct Entry {
		uint64_t tick_rel;
		uint64_t tick_abs;

		Event event;
	};

private:
	Timeline* m_parent;

	std::vector<Entry> m_buf;
	size_t m_num;
	size_t m_cap;
	size_t m_id = 0;

	uint64_t m_t_start;

public:
	static uint64_t GetTick() {
		return rdtsc();
	}

	Timeline(Timeline& parent) {
		m_parent = &parent;
		m_num = 0;
		m_cap = kEventBufSize;
		m_t_start = GetTick();
		m_buf.resize(m_cap);
	}

	void set_id(size_t id) {
		m_id = id;
	}

protected:
	Timeline() {
		m_parent = nullptr;
		m_num = 0;
		m_cap = kEventBufSize;
		m_t_start = GetTick();
		m_buf.resize(m_cap);
	}

public:

	void push(const Event& event) {
		const auto t = GetTick();

		if (m_num >= m_cap) {
			flush();
		}

		// Copy 'event'
		Entry& e = m_buf[m_num];
		e.event = event;
		e.tick_rel = t - m_t_start;
		e.tick_abs = t;
		m_num++;
	}

	virtual void flush() {
		m_parent->bulk_write(m_id, &m_buf[0], m_num);
		m_num = 0;
	}

	virtual void bulk_write(size_t id, const Entry* entries, size_t n) {
		if (m_parent) {
			m_parent->bulk_write(id, entries, n);
		}
	}

	~Timeline() {
		flush();
		m_parent->flush();
	}
};

#include <fstream>
#include <mutex>

template <typename Event>
struct FileTimeline : Timeline<Event> {
private:
	std::mutex m_mutex;

	std::ofstream& m_file;
	bool m_active;

	using Entry = typename Timeline<Event>::Entry;

	static constexpr char *sep = "|";
public:
	FileTimeline(std::ofstream& f) : m_file(f), m_active(true) {
	}

protected:
	virtual void bulk_write(size_t id, const Entry* entries, size_t n) override {
		std::lock_guard<std::mutex> guard(m_mutex);

		if (!m_active) {
			return;
		}

		for (size_t i=0; i<n; i++) {
			const Entry& e = entries[i];
			m_file << id << sep << e.tick_abs << sep << e.tick_rel << sep;
			e.event.serialize(m_file, sep);
			m_file << std::endl;
		}
	}

public:
	virtual void flush() {
		m_file.flush();
	}

	void set_active(bool active) {
		std::lock_guard<std::mutex> guard(m_mutex);

		m_active = active;
	}
};


#endif