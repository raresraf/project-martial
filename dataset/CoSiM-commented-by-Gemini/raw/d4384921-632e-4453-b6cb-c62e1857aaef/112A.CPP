/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem (Codeforces 112A).
 * @details This program performs a case-insensitive lexicographical comparison of
 * two strings and prints -1, 0, or 1 accordingly.
 *
 * @note This implementation uses direct ASCII value manipulation ("magic numbers")
 * for case conversion, which is not portable or standard practice. Using the
 * `tolower()` function from `<cctype>` is generally preferred.
 *
 * @section algorithm
 * Algorithm: In-place Conversion and Direct Comparison
 * 1. Two strings are read from standard input.
 * 2. The code iterates through the strings, converting any uppercase characters
 *    to their lowercase equivalents by adding 32 to their ASCII value.
 * 3. Once both strings are fully lowercase, the standard C++ `string` comparison
 *    operators (`<`, `>`, `==`) are used to efficiently determine their
 *    lexicographical order.
 *
 * @section complexity
 * - Time Complexity: O(L), where L is the length of the strings. This covers
 *   both the linear pass for case conversion and the subsequent comparison.
 * - Space Complexity: O(L), for storing the two input strings.
 */
#include <iostream>
using namespace std;
int main() {
  string a, b;
  cin >> a >> b;

  /**
   * @brief Converts both strings to lowercase in-place using ASCII arithmetic.
   */
  for (int i = 0; i < a.length(); i++) {
    // Checks if the character's ASCII value is in the uppercase range.
    if (a[i] < 92) {
      // Converts to lowercase by adding the ASCII offset (e.g., 'A' (65) + 32 = 'a' (97)).
      a[i] += 32;
    }
    if (b[i] < 92) {
      b[i] += 32;
    }
  }

  /**
   * @brief Performs lexicographical comparison on the normalized strings.
   * This uses the overloaded operators for `std::string`, which is efficient.
   */
  if (a < b) {
    cout << -1;
  } else if (a > b) {
    cout << 1;
  } else if (a == b) {
    cout << 0;
  }
  return 0;
}