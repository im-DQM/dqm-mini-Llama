#include "dqm_llama/terminal.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace dqm {

void Terminal::PrintUserPrompt() const {
  std::cout << "mini-llama > ";
  std::cout.flush();
}

std::string Terminal::ReadLine() const {
  std::string line;
  if (std::getline(std::cin, line)) { return line; }
  return "";
}

void Terminal::PrintAssistantPrefix() const {
    std::cout << "Assistant > ";
  std::cout.flush();
}

void Terminal::PrintTokenText(const std::string& text) const {
  std::cout << text;
}

void Terminal::Flush() const {
  std::cout.flush();
}

void Terminal::NewLine() const {
  std::cout << "\n";
}

void Terminal::PrintHelp() const {
  std::cout << "Commands:\n"
            << "  /help    Show this help message\n"
            << "  /clear   Clear chat history and context\n"
            << "  /stats   Show session statistics\n"
            << "  /params  Show current sampling parameters\n"
            << "  /exit    Exit chat\n";
}

void Terminal::PrintStats(const ChatSession& session) const {
    std::cout << "Session stats:\n"
            << "  messages:           " << session.messages.size() << "\n"
            << "  context tokens:     " << session.token_history.size() << "\n"
            << "  total prompt tokens: " << session.total_prompt_tokens << "\n"
            << "  total generated:     " << session.total_generated_tokens
            << "\n"
            << "  total time:          " << std::fixed << std::setprecision(2)
            << session.total_time_ms << " ms\n"
            << "  tokens/s:            ";
  if (session.total_time_ms > 0.0) {
    std::cout << std::fixed << std::setprecision(2)
              << (session.total_generated_tokens * 1000.0 /
                  session.total_time_ms);
  } else {
    std::cout << "n/a";
  }
  std::cout << std::defaultfloat << "\n";
}

void Terminal::PrintParams(const SamplingParams& params) const {
  std::cout << std::defaultfloat << "Sampling params:\n"
            << "  temperature: " << params.temperature << "\n"
            << "  top_k:       " << params.top_k << "\n"
            << "  seed:        " << params.seed << "\n";
}

void Terminal::PrintMessage(const std::string& msg) const {
  std::cout << msg << "\n";
}

}  // namespace dqm
