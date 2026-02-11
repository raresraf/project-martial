/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 * @details This program provides an idiomatic C++ solution to the "Petya and Strings"
 * problem from Codeforces. It leverages the C++ Standard Library to efficiently
 * compare two strings while ignoring case.
 *
 * @see https://codeforces.com/problemset/problem/112/A
 *
 * Algorithm:
 * 1. Read two strings from standard input.
 * 2. Convert both strings to a consistent case (uppercase in this implementation)
 *    using `std::transform`. This handles the case-insensitivity requirement.
 * 3. Compare the modified strings directly using the overloaded operators for
 *    `std::string`. These operators perform an efficient, character-by-character
 *    lexicographical comparison.
 * 4. Print -1, 1, or 0 based on the result of the comparison.
 *
 * Time Complexity: O(N), where N is the length of the strings. This is determined
 * by the linear-time operations of `std::transform` and the string comparison.
 * Space Complexity: O(N), for storing the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a, b;
  cin >> a >> b;

  // Convert both strings to uppercase for case-insensitive comparison.
  // std::transform applies a function (::toupper) to each element in a range.
  transform(a.begin(), a.end(), a.begin(), ::toupper);
  transform(b.begin(), b.end(), b.begin(), ::toupper);

  // Directly compare the strings lexicographically.
  if (a < b)
    cout << -1 << endl;
  else if (a > b)
    cout << 1 << endl;
  else // This case implies a == b
    cout << 0 << endl;
}
