/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 * @problem 112A - Petya and Strings
 *
 * This program reads two strings, converts them both to lowercase, and then uses
 * the standard string comparison method to determine their lexicographical order.
 * It outputs -1 if the first string is smaller, 1 if it is larger, and 0 if they are equal.
 *
 * @algorithm String Conversion and Comparison
 * @complexity
 * - Time: O(L), where L is the length of the strings. This is dominated by the loop
 *   for converting to lowercase and the subsequent comparison.
 * - Space: O(L) to store the two input strings. The conversion is done in-place.
 */
#include <iostream>
#include <string> // Required for std::string
#include <cctype> // Required for tolower

using namespace std;

int main() {
  string a, b;
  int i, l;

  cin >> a >> b;

  /**
   * @brief Block Logic: Converts both strings to lowercase in-place.
   * Invariant: After this loop, both strings `a` and `b` will consist entirely
   *            of lowercase characters, preserving their original sequence.
   */
  for (i = 0; i < a.size(); i++) {
    a[i] = char(tolower(a[i]));
    b[i] = char(tolower(b[i]));
  }

  /**
   * @brief Functional Utility: Uses the standard library's `compare` method for a
   * robust and efficient lexicographical comparison of the modified strings.
   */
  l = a.compare(b);

  /**
   * @brief Block Logic: Interprets the result of the comparison.
   * The `compare` method returns a negative value if `a` < `b`, a positive value
   * if `a` > `b`, and zero if `a` == `b`.
   */
  if (l < 0)
    cout << "-1";
  else if (l > 0)
    cout << 1;
  else
    cout << 0;

  return 0;
}