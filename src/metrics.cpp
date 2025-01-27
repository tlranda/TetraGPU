#include "metrics.h"

/* Simple class to help with timing things within source code via trivial API.
   It should be as easy as:
        Timer my_timer; // Automatically denotes timer create time, so place
                        // it immediately before your timed section
        my_timer.tick(); // Close the timing interval
   The timer will flush all intervals (including an open interval, if it exists)
   upon its deletion

  You can also do fun things like give text labels to the timer via the label
  functions.
        my_timer.label_prev_interval("testing 123"); // from prior example
        my_timer.label_next_interval("future ooh"); // label in advance of tick open/closed
        my_timer.label_interval(999,"when we get here it will be cool"); // label the 999th interval, why not
  If you want to keep the same timer for a bunch of things, you can give it the
  constructor argument "false" (boolean type) to not immediately start the first
  tick. Just call .tick() when you're ready. The same timer can hold many
  intervals and you don't need to worry about flushing them too much, but you
  can manually flush via .interval(X) with X as:
        * A string, to refer to the previous interval and flush it
        * An integer, to flush the exact interval
  Note that flushing intervals does not delete any data! If you're worried
  about memory you should flush via deleting the timer object or upgrade this
  class to enable content deletion while persisting the object.
*/

// Constructors
Timer::Timer(void) { this->tick(); }
Timer::Timer(bool deferred, std::string name="") {
    if (!deferred) this->tick();
    this->timer_name = name;
}
// Destructor
Timer::~Timer(void) {
    // Print all unannounced closed intervals
    if (this->last_printed < static_cast<int>(this->timings.size()>>1))
        this->all_intervals(this->last_printed+1);
    /*
    else {
        std::cerr << "Deleting timer with " << (this->timings.size()>>1)
                  << " intervals; " << this->last_printed << " have been printed"
                  << std::endl;
    }
    */
}

void Timer::set_timer_name(std::string name) { this->timer_name = name; }

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
        std::cerr << EXCLAIM_EMOJI << "Timer[" << this->timer_name << "] ";
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
        std::cout << "Timer[" << this->timer_name << "] "
                  << "Elapsed time for interval " << idx << "(" << (idx<<1)
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
        std::cout << EXCLAIM_EMOJI << "Timer[" << this->timer_name << "] "
                  << "Open interval ";
        auto label = this->labels.find(this->open_interval);
        if (label == this->labels.end()) std::cout << this->open_interval;
        else std::cout << label->second;
        std::cout << " cannot report time until it is closed by another tick()"
                  << std::endl;
    }
}

