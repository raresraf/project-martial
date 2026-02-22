/**
 * @file 112A.CPP
 * @brief Solution for the "Petya and Strings" problem (Codeforces 112A).
 * @see https://codeforces.com/problemset/problem/112/A
 *
 * @algorithm
 * The problem requires a case-insensitive lexicographical comparison of two strings.
 * The output should be -1 if the first string is less than the second, 1 if it is
 * greater, and 0 if they are equal.
 *
 * This solution uses a straightforward "convert-then-compare" approach:
 * 1. Read the two input strings.
 * 2. Iterate through both strings simultaneously, converting each character to
 *    lowercase. This ensures the subsequent comparison is case-insensitive.
 * 3. After conversion, use the standard C++ string comparison operators (`>`, `==`)
 *    to determine the lexicographical relationship between the two modified strings.
 * 4. Print -1, 1, or 0 based on the comparison result.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the strings. This includes
 *   the time to read the strings, convert them to lowercase, and then compare them.
 * - Space Complexity: O(L) to store the two input strings.
 */
// https://codeforces.com/problemset/problem/112/A
// 112 A. Petya and Strings

#include <iostream>
// Note: Including <string> and <cctype> is good practice.
using namespace std;
int main() {
  string a, b;
  cin >> a >> b;
  /**
   * @brief Convert both strings to lowercase for case-insensitive comparison.
   */
  for (int i = 0; i < a.size(); i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }
  /**
   * @brief Perform the lexicographical comparison on the lowercase strings and
   * print the corresponding result (-1, 0, or 1).
   */
  if (a > b)
    cout << 1;
  else if (a == b)
    cout << 0;
  else // (a < b)
    cout << -1;
}
