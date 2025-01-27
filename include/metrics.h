#ifndef TETRA_METRICS
#define TETRA_METRICS

#include <vector>
#include <map>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

#include "emoji.h"

typedef std::chrono::time_point<std::chrono::system_clock> ClockReading;
#define get_clock std::chrono::system_clock::now

#define time_cast(type, amount) std::chrono::duration_cast<type>(amount)
#define seconds(amount) time_cast(std::chrono::seconds, amount).count()
#define microseconds(amount) time_cast(std::chrono::microseconds, amount).count() % 1'000'000

class Timer {
    /*
     * Very basic timer class that can be spun up once to count a bunch of
     * intervals. You can also label the intervals to make the output easier
     * to parse.
     */
    private:
        // Stores timestamps to later become intervals
        std::vector<ClockReading> timings;
        // Optional mapping of interval indices to strings
        std::map<int,std::string> labels;
        // Prefix name for outputs
        std::string timer_name;
        // Tracks what has been printed
        int last_printed = -1;
        // Convenient way to track INTERVALS rather than vector entries,
        // automatically controlled by tick()
        int open_interval = 0;
    public:
        // Creation / Destruction / Copy
        Timer(void);
        Timer(bool deferred, std::string name);
        ~Timer(void);

        // Rename timer
        void set_timer_name(std::string name);

        // Log a timestamp
        void tick(void);
        void tick_announce(void);

        // Label controls
        void label_interval(int idx, std::string label);
        void label_prev_interval(std::string label);
        void label_next_interval(std::string label);

        // Outputs
        void interval(int idx = -1);
        void interval(std::string label);
        void all_intervals(int idx = 0);
};

#endif

