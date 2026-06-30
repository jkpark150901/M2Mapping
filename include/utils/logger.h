#pragma once
// Minimal thread-safe, level-based logger writing to its own file (+ console).
// Header-only, no extra deps. Independent of llog (which is value/timer logging).
//
// Usage:
//   #include "utils/logger.h"
//   logger::init(k_output_path / "debug.log", logger::Level::DEBUG); // once at startup
//   LOG_INFO("views=" << V << " out_dim=" << dim);
//   LOG_DEBUG("sample took " << ms << " ms");
//   LOG_WARN("no coverage for " << n << " pts");
//
// Level threshold: only messages >= current level are emitted.
// Set via init(...) or env var  M2_LOG_LEVEL = debug|info|warn|error.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace logger {

enum class Level { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

class Logger {
public:
  static Logger &instance() {
    static Logger inst;
    return inst;
  }

  void init(const std::filesystem::path &path, Level level = Level::INFO,
            bool also_console = true) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!path.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(path.parent_path(), ec);
      ofs_ = std::ofstream(path); // truncate on init
    }
    level_ = level_from_env(level);
    console_ = also_console;
    ready_ = true;
  }

  void set_level(Level l) {
    std::lock_guard<std::mutex> lk(mtx_);
    level_ = l;
  }

  void log(Level lv, const char *file, int line, const std::string &msg) {
    if (lv < level_)
      return;
    std::lock_guard<std::mutex> lk(mtx_);
    std::ostringstream os;
    os << ts() << " [" << name(lv) << "] " << basename(file) << ':' << line
       << "  " << msg << '\n';
    const std::string s = os.str();
    if (ofs_.is_open()) {
      ofs_ << s;
      ofs_.flush();
    }
    if (console_)
      (lv >= Level::WARN ? std::cerr : std::cout) << s;
  }

private:
  Logger() = default;

  static const char *name(Level l) {
    switch (l) {
    case Level::DEBUG: return "DEBUG";
    case Level::INFO:  return "INFO ";
    case Level::WARN:  return "WARN ";
    default:           return "ERROR";
    }
  }
  static std::string basename(const char *f) {
    std::string s(f);
    auto p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
  }
  static std::string ts() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream o;
    o << std::put_time(std::localtime(&t), "%H:%M:%S");
    return o.str();
  }
  static Level level_from_env(Level fallback) {
    const char *e = std::getenv("M2_LOG_LEVEL");
    if (!e)
      return fallback;
    std::string v(e);
    for (auto &c : v) c = std::tolower(c);
    if (v == "debug") return Level::DEBUG;
    if (v == "info")  return Level::INFO;
    if (v == "warn")  return Level::WARN;
    if (v == "error") return Level::ERROR;
    return fallback;
  }

  std::ofstream ofs_;
  std::mutex mtx_;
  Level level_ = Level::INFO;
  bool console_ = true;
  bool ready_ = false;
};

inline void init(const std::filesystem::path &path, Level level = Level::INFO,
                 bool also_console = true) {
  Logger::instance().init(path, level, also_console);
}
inline void set_level(Level l) { Logger::instance().set_level(l); }

} // namespace logger

// ---- macros (lazy: stream is only built if level passes is checked inside) ----
#define LOG_AT(lv, expr)                                                        \
  do {                                                                          \
    std::ostringstream _los;                                                    \
    _los << expr;                                                               \
    ::logger::Logger::instance().log((lv), __FILE__, __LINE__, _los.str());     \
  } while (0)

#define LOG_DEBUG(expr) LOG_AT(::logger::Level::DEBUG, expr)
#define LOG_INFO(expr) LOG_AT(::logger::Level::INFO, expr)
#define LOG_WARN(expr) LOG_AT(::logger::Level::WARN, expr)
#define LOG_ERROR(expr) LOG_AT(::logger::Level::ERROR, expr)
