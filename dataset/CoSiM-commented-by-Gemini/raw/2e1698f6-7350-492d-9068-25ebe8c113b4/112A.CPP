/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem, performing case-insensitive string comparison.
 * @details This program reads two strings, converts them both to lowercase, and then determines their
 * lexicographical order.
 *
 * Algorithm:
 * 1. Read two input strings, `a` and `b`.
 * 2. Convert both strings entirely to lowercase in-place by iterating through them.
 * 3. First, perform a direct comparison for equality (`a == b`). If they are identical, print 0.
 * 4. If they are not equal, iterate through the strings again, character by character.
 * 5. At the first position where the characters differ, determine which string is lexicographically smaller.
 *    - If the character in `a` is less than the character in `b`, print -1.
 *    - If the character in `a` is greater than the character in `b`, print 1.
 * 6. Once the first difference is found and the result is printed, the loop is terminated.
 *
 * @note This implementation is slightly inefficient because it performs two passes for comparison in the non-equal case:
 * one for the `if (a == b)` check and another in the `for` loop. A more streamlined approach would be to
 * convert to lowercase and then use a single `if (a < b)` / `else if (a > b)` structure.
 *
 * Time Complexity: O(L), where L is the length of the strings.
 * Space Complexity: O(1), as the strings are modified in-place.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  string a, b;
  int i;
  cin >> a >> b;

  /**
   * Block Logic: Convert both strings to lowercase in-place.
   * Pre-condition: `a` and `b` hold the original input strings.
   * Invariant: After the loop, `a` and `b` will be entirely lowercase.
   */
  for (i = 0; i < a.size(); i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }

  // First, check for overall equality after normalization.
  if (a == b)
    cout << "0" << endl;
  else {
    /**
     * Block Logic: Find the first differing character to determine lexicographical order.
     * This manual comparison is correct but redundant, as `if (a < b)` would achieve the same result more concisely.
     */
    for (i = 0; i < a.size(); i++) {
      if (a[i] < b[i]) {
        cout << "-1" << endl;
        break; // Exit after finding the first difference.
      }
      if (a[i] > b[i]) {
        cout << "1" << endl;
        break; // Exit after finding the first difference.
      }
    }
  }
}
