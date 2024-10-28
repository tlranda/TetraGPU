#include "metrics.h"

// Constructors
Timer::Timer(void) { this->tick(); }
Timer::Timer(bool deferred) { if (!deferred) this->tick(); }
// Destructor
Timer::~Timer(void) {
    // Print all unannounced closed intervals
    if (this->last_printed < (this->timings.size()<<1))
        this->all_intervals(this->last_printed+1);
}


// Log timestamp
void Timer::tick(void) {
    this->timings.emplace_back(get_clock());
    // Closed an interval
    if (this->timings.size() % 2 == 0) this->open_interval++;
}
void Timer::tick_announce(void) {
    this->tick();
    this->interval();
}


// Label controls
void Timer::label_interval(int idx, std::string label) {
    this->labels[idx] = label;
}
void Timer::label_prev_interval(std::string label) {
    // Use interval BEHIND current open interval
    this->labels[this->open_interval-1] = label;
}
void Timer::label_next_interval(std::string label) {
    // Use current open interval
    this->labels[this->open_interval] = label;
}

// Outputs
void Timer::interval(int idx) {
    // Kick to lastest closed interval when default arguments (see .h) used
    if (idx < 0) idx = this->open_interval - 1;

    // Look for a defined label
    auto label = this->labels.find(idx);

    // Early-exit: Interval is not closed
    if (idx >= this->open_interval) {
        std::cerr << EXCLAIM_EMOJI;
        if (label == this->labels.end()) std::cerr << "Interval " << idx;
        else std::cerr << label->second;
        std::cerr << " cannot be reported until it is closed" << std::endl;
        return;
    }

    // Normal exit: Display duration with appropriate label
    // Bit-shift interval idx (and bit-or +1 for second point) to get timing indices
    ClockReading latest = this->timings[(idx<<1|1)],
                 prior  = this->timings[(idx<<1)];
    auto elapsed = latest - prior;
    auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    auto elapsed_subsec = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    elapsed_subsec = elapsed_subsec % 1'000'000;

    std::cout << HOURGLASS_EMOJI;
    if (label == this->labels.end())
        std::cout << "Elapsed time for interval " << idx << "(" << (idx<<1)
                  << ", " << (idx<<1|1) << ")" << ": ";
    else std::cout << label->second << ": ";
    std::cout << elapsed_sec << "." << std::setfill('0') << std::setw(6)
              << elapsed_subsec << std::endl;

    // Update last printed
    this->last_printed = idx;
}

void Timer::interval(std::string label) {
    this->label_prev_interval(label);
    this->interval(-1);
}

void Timer::all_intervals(int idx) {
    for ( ; idx < this->open_interval; idx++)
        this->interval(idx);
    if (this->timings.size() % 2) {
        std::cout << EXCLAIM_EMOJI << "Open interval ";
        auto label = this->labels.find(this->open_interval);
        if (label == this->labels.end()) std::cout << this->open_interval;
        else std::cout << label->second;
        std::cout << " cannot report time until it is closed by another tick()"
                  << std::endl;
    }
}

