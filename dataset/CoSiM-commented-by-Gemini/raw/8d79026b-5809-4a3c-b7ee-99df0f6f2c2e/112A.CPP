/**
 * @file 112A.CPP
 * @brief Solution for case-insensitive lexicographical string comparison.
 * @details This program compares two input strings lexicographically, ignoring differences in case.
 * It outputs -1 if the first string is smaller, 1 if it is larger, and 0 if they are equal.
 * This is a common problem in introductory competitive programming (e.g., Codeforces 112A "Petya and Strings").
 *
 * @algorithm Case-insensitive linear string comparison.
 * @complexity
 * - Time: O(L), where L is the length of the strings. The complexity is dominated by the
 *   `transform` operations and the final comparison loop, both of which are linear in L.
 * - Space: O(L) to store the two input strings.
 */
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
  string a, b;
  cin >> a >> b;

  /**
   * @block Case Normalization
   * @logic Converts both strings to lowercase to ensure the comparison is case-insensitive.
   * `std::transform` applies the `::tolower` function to each character in the string.
   */
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);

  /**
   * @block Lexicographical Comparison
   * @logic Iterates through the characters of both strings to find the first point of difference.
   */
  for (int i = 0; i < a.length(); i++) {
    char ac = a.at(i);
    char bc = b.at(i);
    if (ac != bc) {
      // If a character in string 'a' is greater than in 'b', 'a' is lexicographically larger.
      if (ac > bc) {
        printf("1
");
      } else {
        // Otherwise, 'a' is lexicographically smaller.
        printf("-1
");
      }
      // Terminate after finding the first difference.
      return 0;
    }
  }

  // If the loop completes without returning, the strings are identical.
  printf("0
");
  return 0;
}
