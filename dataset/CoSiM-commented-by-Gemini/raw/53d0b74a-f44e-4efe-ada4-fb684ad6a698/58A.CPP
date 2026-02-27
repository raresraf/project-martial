/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem from Codeforces using regular expressions.
 * @details The problem is to determine if a given string contains "hello" as
 * a subsequence, meaning the characters 'h', 'e', 'l', 'l', 'o' appear in
 * order, but not necessarily contiguously.
 *
 * @algorithm
 * This solution uses a C++ regular expression to elegantly solve the problem.
 * 1. A regex pattern is defined to match the subsequence "hello". The pattern is:
 *    `^[a-z]*h[a-z]*e[a-z]*l[a-z]*l[a-z]*o[a-z]*$`
 *    - `^` and `$` anchor the match to the start and end of the string.
 *    - `h`, `e`, `l`, `l`, `o` match the literal characters of the subsequence.
 *    - `[a-z]*` matches any sequence of zero or more lowercase letters,
 *      effectively allowing any number of other characters between the letters
 *      of "hello".
 * 2. The input string is read.
 * 3. `std::regex_match` is used to check if the entire input string matches
 *    the defined regex pattern.
 * 4. If it matches, "hello" is a subsequence, and "YES" is printed. Otherwise,
 *    "NO" is printed.
 *
 * @complexity
 * Time Complexity: The complexity of `std::regex_match` can vary, but for this
 * specific non-backtracking pattern, it is typically linear, O(L), where L is
 * the length of the input string.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <regex>
#include <string>

int main() {
  std::string s; // The input string.
  
  /**
   * @brief A regular expression to find "hello" as a subsequence.
   * The pattern looks for 'h', then 'e', then 'l', etc., in order,
   * allowing any number of other lowercase characters ([a-z]*) in between.
   * `^` and `$` ensure the entire string is consumed by the pattern.
   */
  std::regex e("^[a-z]*h[a-z]*e[a-z]*l[a-z]*l[a-z]*o[a-z]*$");

  std::cin >> s;

  // Use regex_match to test if the input string conforms to the subsequence pattern.
  std::cout << (std::regex_match(s, e) ? "YES" : "NO") << std::endl;

  return 0;
}
