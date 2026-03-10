/**
 * @file 58A.CPP
 * @brief Solution to a competitive programming problem checking for a subsequence.
 * This program determines if the word "hello" can be formed by deleting zero or
 * more characters from a given input string.
 *
 * @algorithm Greedy linear scan for subsequence checking.
 * @time_complexity O(N), where N is the length of the input string.
 * @space_complexity O(L), where L is the maximum allowed length of the input string.
 */
#include <iostream>
using namespace std;
int main() {
  int i, j = 10, k = 0, n = 0;
  char a[101], b[] = "hello";
  cin >> a;
  /**
   * @block
   * @brief Scans the input string to find the "hello" subsequence.
   *
   * This loop iterates through the input string `a` and greedily matches
   * characters with the target string `b` ("hello"). The variable `n` tracks
   * the current character to find in "hello", and `k` counts the number of
   * matched characters.
   */
  for (i = 0; a[i] != '\0'; i++) {
    // If the current character in the input matches the target character
    if (a[i] == b[n]) {
      // Increment match count and advance to the next target character
      k++;
      n++;
    }
    // If all 5 characters of "hello" have been found in order
    if (k == 5) {
      cout << "YES";
      return 0;
    }
  }
  // If the loop completes and not all characters were found
  if (k != 5)
    cout << "NO";
  return 0;
}
