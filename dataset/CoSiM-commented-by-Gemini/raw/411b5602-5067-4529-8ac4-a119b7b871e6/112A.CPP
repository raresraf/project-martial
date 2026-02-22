/**
 * @file 112A.CPP
 * @brief Solution to Codeforces problem 112A - Petya and Strings.
 *
 * @algorithm
 * The problem requires a case-insensitive lexicographical comparison of two strings.
 * This solution performs a character-by-character comparison.
 *
 * The algorithm is as follows:
 * 1. Read two strings from standard input.
 * 2. Initialize a flag `f` to 0, which will store the result (0 for equal,
 *    -1 for less, 1 for greater).
 * 3. Iterate through both strings simultaneously using a single loop.
 * 4. In each iteration, take the characters at the current position from both strings.
 * 5. Manually convert both characters to uppercase for a case-insensitive comparison.
 *    This is done by checking their ASCII values against the range for lowercase
 *    letters (97-122) and subtracting 32 if they fall in that range.
 * 6. Compare the uppercase characters. If they are different, set the flag `f` to
 *    -1 or 1 accordingly and exit the loop immediately.
 * 7. If the loop completes without any differences, the flag `f` remains 0.
 * 8. Print the final value of the flag.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the strings. The loop runs at most
 *   L times and may terminate early.
 * - Space Complexity: O(L) to store the two input strings.
 */
#include <iostream>
#include <math.h> // This header is not used.
using namespace std;

int main() {

  char ch1, ch2;
  string s1, s2;
  int f = 0; // Flag: 0 for equal, -1 for s1<s2, 1 for s1>s2

  getline(cin, s1);
  getline(cin, s2);

  /**
   * @brief Iterates through the strings, performing a case-insensitive
   * character-by-character comparison.
   */
  for (int i = 0; i < s1.length(); ++i) {
    ch1 = s1[i];
    ch2 = s2[i];
    // Manually convert the character from s1 to uppercase.
    // 97 is 'a', 122 is 'z'. Subtracting 32 converts to uppercase.
    if ((int)s1[i] >= 97 && (int)s1[i] <= 122) {
      ch1 = (int)s1[i] - 32;
    }
    // Manually convert the character from s2 to uppercase.
    if ((int)s2[i] >= 97 && (int)s2[i] <= 122) {
      ch2 = (int)s2[i] - 32;
    }
    /**
     * @brief Compares the two uppercase characters.
     * If a difference is found, set the result flag and break the loop
     * as the lexicographical order has been determined.
     */
    if ((int)ch1 < (int)ch2) {
      f = -1;
      break;
    } else if ((int)ch1 > (int)ch2) {
      f = 1;
      break;
    }
  }

  cout << f;

  return 0;
}
