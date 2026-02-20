/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings and compares them lexicographically, treating
 * uppercase and lowercase letters as equivalent. It outputs -1 if the first
 * string is lexicographically smaller than the second, 1 if it is larger,
 * and 0 if they are equal.
 *
 * Algorithm:
 * 1. Read two strings from standard input.
 * 2. Convert both strings to lowercase.
 * 3. Perform a lexicographical comparison on the modified strings.
 * 4. Print -1, 1, or 0 based on the comparison result.
 *
 * Time Complexity: O(L), where L is the length of the strings. The conversion
 * to lowercase and the comparison both take linear time.
 * Space Complexity: O(L) to store the two strings.
 */
#include <algorithm>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string first, second;
  // Read the two strings to be compared.
  cin >> first >> second;

  /**
   * Block Logic: Case Normalization
   * The `transform` function is used to apply the `tolower` operation to every
   * character in both strings. This ensures that the subsequent comparison is
   * case-insensitive. The C-style cast `(int (*)(int))tolower` is used to
   * resolve ambiguity for the overloaded `tolower` function.
   */
  transform(first.begin(), first.end(), first.begin(), (int (*)(int))tolower);
  transform(second.begin(), second.end(), second.begin(),
            (int (*)(int))tolower);

  /**
   * Block Logic: Lexicographical Comparison
   * The overloaded `>` and `<` operators for `std::string` perform a
   * lexicographical comparison.
   * - If `first` is lexicographically greater than `second`, print "1".
   * - If `first` is lexicographically less than `second`, print "-1".
   * - If they are equal, print "0".
   */
  if (first > second) {
    cout << "1";
  } else if (first < second) {
    cout << "-1";
  } else
    cout << "0";
  return 0;
}
