/**
 * @file 58A.CPP
 * @brief Logic for greedy subsequence detection within a character stream.
 * 
 * Logic: Determines if the target sequence "hello" exists as a subsequence in 
 * the input string. The characters 'h', 'e', 'l', 'l', 'o' must appear in order, 
 * potentially interspersed with other characters.
 * 
 * Algorithm: Single-pass scan with a monotonic pattern pointer.
 * Time Complexity: $O(N)$ where N is the length of the input string.
 * Space Complexity: $O(N)$ for the input buffer.
 */

#include <iostream>
#include <string>

/**
 * @brief Functional Utility: Standardized boolean-to-string result mapper.
 * @param v The success state of the subsequence search.
 */
void answer(bool v) {
  constexpr const char *s[2] = {"NO", "YES"};
  std::cout << s[v] << '\n';
}

/**
 * @brief core search logic for the "hello" pattern.
 * @param input The source character sequence.
 */
void solve(const std::string &input) {
  // Target pattern representing the success state.
  constexpr char m[] = "hello";

  // State: Index of the next character in the pattern to be matched.
  size_t pattern_index = 0;

  /**
   * Block Logic: Greedily consumes input characters to match the pattern.
   * Invariant: pattern_index represents the length of the longest prefix 
   * of "hello" found so far as a subsequence.
   */
  for (const char c : input) {
    /**
     * Logic: If current character matches the target, advance the pattern pointer.
     * Termination: Succeeds immediately if the 5th character ('o') is matched.
     */
    if (c == m[pattern_index] && ++pattern_index == 5)
      return answer(true);
  }

  // Fallback: Pattern not fully identified.
  answer(false);
}

int main() {
  std::string input;
  // Reads the raw user input.
  if (!(std::cin >> input)) return 0;

  solve(input);

  return 0;
}
