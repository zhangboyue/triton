#ifndef TRITON_TOOLS_BENCH_HPP
#define TRITON_TOOLS_BENCH_HPP

namespace triton{
namespace tools{

class timer{
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::nanoseconds nanoseconds;

public:
    explicit timer(bool run = false)
    { if (run) start(); }

    void start()
    { _start = high_resolution_clock::now(); }

    nanoseconds get() const
    { return std::chrono::duration_cast<nanoseconds>(high_resolution_clock::now() - _start); }

private:
    high_resolution_clock::time_point _start;
};

template<class OP, class SYNC>
double bench(OP const & op, SYNC const & sync, const triton::driver::device * device)
{
  timer tmr;
  std::vector<size_t> times;
  double total_time = 0;
  op();
  sync();
  while(total_time*1e-9 < 1e-3){
    float norm = 1;
    // normalize clock if possible to get roughly constant result
    if(auto cu_device = dynamic_cast<const triton::driver::cu_device*>(device))
      norm = (float)cu_device->current_sm_clock()/cu_device->max_sm_clock();
    tmr.start();
    op();
    sync();
    times.push_back(norm*tmr.get().count());
    total_time+=times.back();
  }
  return *std::min_element(times.begin(), times.end());
}

}
}

#endif
