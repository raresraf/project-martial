/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings and compares them lexicographically, ignoring
 * case differences. It outputs -1 if the first string is smaller, 1 if it is
 * larger, and 0 if they are equal. This version first converts both strings
 * to uppercase and then performs the comparison.
 *
 * Algorithm:
 * 1. Read two strings, `a` and `b`.
 * 2. Use the `std::transform` function to convert string `a` to uppercase in-place.
 * 3. Similarly, convert string `b` to uppercase.
 * 4. Initialize an integer `x` to 0, which will store the comparison result.
 * 5. Iterate through the characters of the now-uppercase strings.
 * 6. If a character in `a` is greater than the corresponding character in `b`,
 *    set `x` to 1 and break the loop.
 * 7. If a character in `a` is less than the corresponding character in `b`,
 *    set `x` to -1 and break the loop.
 * 8. If the loop completes without finding any differences, `x` remains 0.
 * 9. Print the final value of `x`.
 *
 * Time Complexity: O(N), where N is the length of the strings. The process involves
 * two passes of O(N) for the transformations and one pass for the comparison.
 * Space Complexity: O(N) to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string a, b;
  cin >> a >> b;
  
  // Convert both strings to uppercase for case-insensitive comparison.
  transform(a.begin(), a.end(), a.begin(), ::toupper);
  transform(b.begin(), b.end(), b.begin(), ::toupper);

  int x = 0; // Stores the result: -1, 0, or 1.
  
  /**
   * @brief Loop through the strings to find the first differing character.
   * Pre-condition: Both strings `a` and `b` are in uppercase.
   */
  for (int i = 0; i < a.length(); i++) {
    if ((int)a[i] > (int)b[i]) {
      x = 1;
      break;
    } else if ((int)a[i] < (int)b[i]) {
      x = -1;
      break;
    }
  }
  
  // Print the result of the lexicographical comparison.
  cout << x << endl;
}