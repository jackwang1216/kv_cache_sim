#pragma once
#include <cstdint>

enum class EventType {
    Arrival,
    Enqueue,
    StartPrefill,
    StartDecode,
    Finish,
    Reject,
    Evict
};

struct Event {
    double time_ms = 0.0;
    EventType type = EventType::Arrival;
    int request_index = -1;
    int extra = 0;
};

struct EventCompare {
    bool operator()(const Event& a, const Event& b) const {
        return a.time_ms > b.time_ms;
    }
};
