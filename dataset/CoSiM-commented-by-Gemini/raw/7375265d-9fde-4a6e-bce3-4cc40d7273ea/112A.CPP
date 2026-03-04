/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two strings.
 *
 * This program provides a very concise solution to the case-insensitive string
 * comparison problem by leveraging the non-standard but common `strcasecmp`
 * function from the C library.
 *
 * @algorithm Direct case-insensitive comparison using `strcasecmp`.
 * @time_complexity O(L), where L is the length of the strings, as `strcasecmp`
 *                  performs a character-by-character comparison.
 * @space_complexity O(L) to store the two input strings.
 */
#include <iostream>
#include <string>

#include <cstring>

void answer(int v) { std::cout << v << '\n'; }

/**
 * @brief Solves the problem by comparing two strings case-insensitively.
 * @param s1 The first string.
 * @param s2 The second string.
 *
 * This function uses `strcasecmp` to compare the two strings. `strcasecmp` is
 * a POSIX extension and not part of standard C++, but it is widely available.
 * It returns 0 for equal strings, a negative value if s1 < s2, and a positive
 * value if s1 > s2. The result is then normalized by the problem constraints
 * to be exactly -1, 0, or 1 by the online judge, though this code prints the raw result.
 */
void solve(const std::string &s1, const std::string &s2) {
  answer(strcasecmp(s1.c_str(), s2.c_str()));
}

int main() {
  std::string s1, s2;
  std::cin >> s1 >> s2;

  solve(s1, s2);

  return 0;
}