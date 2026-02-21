/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * @details This program reads two strings and compares them character by character, ignoring case. 
 * It outputs -1 if the first string is lexicographically smaller, 1 if it is larger, 
 * and 0 if they are equal. This is a common task in programming contests.
 *
 * @algorithm The program implements a custom case-insensitive string comparison. It iterates
 * through the strings, converting each character to lowercase using `tolower()` before
 * comparison. The result (-1, 1, or 0) is determined by the first pair of
 * characters that do not match.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the strings, as it involves a single pass.
 * Space Complexity: O(N), for storing the two input strings.
 */
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string line1, line2;
  cin >> line1 >> line2;

  // This check is likely redundant under typical problem constraints where strings are of equal length.
  if (line1.size() != line2.size()) {
    return -1;
  }

  /**
   * @brief Block Logic: Iterates through the strings to find the first lexicographical difference.
   * @invariant At the beginning of each iteration `i`, the prefixes `line1[0...i-1]` and 
   * `line2[0...i-1]` are equal when ignoring case.
   */
  for (decltype(line1.size()) i = 0; i != line1.size(); ++i) {
    // Compares characters at the current position, ignoring case.
    if (tolower(line1[i]) < tolower(line2[i])) {
      cout << "-1";
      return 0;
    } else if (tolower(line1[i]) > tolower(line2[i])) {
      cout << "1";
      return 0;
    }
  }

  // If the loop completes, the strings are equal (case-insensitively).
  cout << "0";
  return 0;
}
