#pragma once

#include <chrono>
#include <cstdio>


class Timer {
public:
    Timer(int64_t* duration_ptr)
        : m_duration_ptr{duration_ptr},
          m_start_time{std::chrono::high_resolution_clock::now()}
    { 
    }
    ~Timer() { stop(); }

    void stop() {
        if (m_stopped) {
            return;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        int64_t start = std::chrono::time_point_cast<std::chrono::milliseconds>(m_start_time).time_since_epoch().count();
        int64_t end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();
        int64_t duration = end - start;
        *m_duration_ptr += duration;
        m_stopped = true;
    }

private:
    int64_t* m_duration_ptr;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
    bool m_stopped = false;
};
