/**
 * @file 112A.CPP
 * @brief Solution to Codeforces Problem 112A: "Petya and Strings".
 *        The problem involves comparing two strings lexicographically, ignoring case.
 *        The output should be -1 if the first string is lexicographically smaller,
 *        1 if the second string is lexicographically smaller, and 0 if they are equal.
 *
 * Algorithm: String transformation and comparison. Both input strings are converted
 *            to a consistent case (lowercase in this solution). Then, a standard
 *            lexicographical comparison function (like `strcmp` after converting to C-style strings)
 *            is used to determine their order.
 * Time Complexity: O(L) where L is the maximum length of the two strings, due to
 *                  two passes for transformation and one pass for comparison.
 * Space Complexity: O(L) to store the modified strings.
 */
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  /**
   * Functional Utility: Declaration of string variables for input.
   * `s1`: Stores the first input string.
   * `s2`: Stores the second input string.
   */
  string s1, s2;
  // Input: Read the two strings from standard input.
  cin >> s1 >> s2;

  /**
   * Block Logic: Convert both strings to lowercase for case-insensitive comparison.
   * Algorithm: `std::transform` applies the `::tolower` function to each character
   *            in the range `[s.begin(), s.end())` and stores the result back
   *            into the same string, effectively modifying it in-place.
   */
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  /**
   * Block Logic: Compare the two modified strings lexicographically.
   * Functional Utility: `strcmp` compares two C-style strings. `s.c_str()`
   *            returns a pointer to a null-terminated character array
   *            representing the string's content.
   * Return Value:
   *   - A negative value if `s1` is lexicographically less than `s2`.
   *   - A positive value if `s1` is lexicographically greater than `s2`.
   *   - Zero if `s1` and `s2` are lexicographically equal.
   */
  cout << strcmp(s1.c_str(), s2.c_str()) << endl;
  return 0;
}
