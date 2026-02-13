/**
 * @file 112A.CPP
 * @brief Solution for the "Petya and Strings" problem.
 * @details This program compares two strings lexicographically, ignoring case. It reads two
 * strings, converts them both to lowercase, and then determines their lexical order,
 * printing -1, 1, or 0 accordingly.
 *
 * @section algorithm Algorithm
 * 1. Read two strings, `a` and `b`, from standard input.
 * 2. Iterate through both strings character by character from beginning to end.
 * 3. In each iteration, convert the characters at the current position in both strings
 *    to lowercase. This is done by checking if the character's ASCII value is less
 *    than 92 (a proxy for being an uppercase letter) and, if so, adding 32 to it.
 * 4. After this normalization, the two strings (now entirely in lowercase) are
 *    compared lexicographically using standard C++ string operators.
 * 5. If string `a` is less than `b`, print -1.
 * 6. If string `a` is greater than `b`, print 1.
 * 7. If they are equal, print 0.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the strings. The process involves a single pass for case conversion and another for comparison.
 * - Space Complexity: O(L) to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a, b;
  cin >> a >> b;

  /**
   * @brief This loop iterates through the strings to convert them to lowercase
   * for a case-insensitive comparison.
   */
  for (int i = 0; i < a.size(); ++i) {
    // The ASCII value for 'Z' is 90. Checking for < 92 is a way to identify
    // uppercase English letters without using library functions like `is_upper`.
    if (a[i] < 92) {
      a[i] += 32; // Adding 32 converts an uppercase letter to its lowercase equivalent in ASCII.
    }
    if (b[i] < 92) {
      b[i] += 32;
    }
  }

  // After converting both strings to lowercase, perform a standard lexicographical comparison.
  if (a < b) {
    cout << -1;
  } else if (a > b) {
    cout << 1;
  } else if (a == b) {
    cout << 0;
  }
}
