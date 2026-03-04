/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A - "Football".
 *
 * This program checks if a given string of '0's and '1's contains a
 * "dangerous" situation, defined as 7 or more consecutive identical characters.
 *
 * @algorithm Linear scan with running counters.
 * @time_complexity O(N), where N is the length of the string.
 * @space_complexity O(N) to store the input string.
 * @note The loop condition `i <= s.size()` is a potential off-by-one error,
 *       as it will access one character beyond the end of the string.
 */
#include <iostream>

int main() {
  int nb0 = 0, nb1 = 0;
  ;
  std::string s;
  std::cin >> s;
  /**
   * @brief Scans the string for 7 consecutive identical characters.
   * This loop iterates through the string, using `nb0` and `nb1` to count
   * consecutive '0's and '1's. When a character is found, the corresponding
   * counter is incremented and the other is reset. If either counter
   * reaches 7, the condition is met, "YES" is printed, and the program exits.
   */
  for (int i = 0; i <= s.size(); i++) {
    if (s[i] == '1') {
      nb1++;
      nb0 = 0;
    }
    if (s[i] == '0') {
      nb1 = 0;
      nb0++;
    }
    if (nb0 >= 7 or nb1 >= 7) {
      std::cout << "YES";
      return 0;
    }
  }
  // If the loop completes without finding 7 consecutive characters.
  std::cout << "NO";
}