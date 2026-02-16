/**
 * @file 112A.CPP
 * @brief Solution for Codeforces problem 112A, "Petya and Strings".
 *
 * @description
 * This program performs a case-insensitive lexicographical comparison of two
 * input strings. It prints -1 if the first string is smaller, 1 if it is
 * larger, and 0 if they are equal.
 *
 * This implementation is a clean and idiomatic C++ approach, using the
 * `std::transform` algorithm to normalize the strings before comparison.
 *
 * @algorithm
 * 1. Read two `std::string`s, s1 and s2.
 * 2. Use `std::transform` with `::tolower` to convert the entire `s1` string
 *    to lowercase, in-place.
 * 3. Repeat the process for `s2`.
 * 4. With both strings normalized to lowercase, perform a direct lexicographical
 *    comparison using the overloaded `<`, `>`, and `==` operators.
 * 5. Print -1, 1, or 0 based on the comparison result.
 *
 * @time_complexity O(L), where L is the length of the strings, as `transform`
 *   and the final comparison are both linear operations.
 * @space_complexity O(L) to store the two strings.
 */
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s1, s2;
  cin >> s1 >> s2;
  // The 'flag' variable is declared but never used.
  int flag = 0;

  // Block Logic: Normalize both strings to lowercase for case-insensitive comparison.
  // `std::transform` applies the `::tolower` function to each character in the string.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  // Block Logic: Perform lexicographical comparison on the normalized strings.
  if (s1 < s2) {
    cout << -1;
  } else if (s1 > s2) {
    cout << 1;
  } else {
    cout << 0;
  }
  return 0;
}
