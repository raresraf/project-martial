#include <bits/stdc++.h>
using namespace std;

/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem from Codeforces (Problem 112A).
 * @details This program performs a case-insensitive lexicographical comparison of two input strings.
 *
 * Algorithm:
 * The program reads two strings from standard input. To make the comparison case-insensitive,
 * both strings are converted to lowercase using the `std::transform` function with `::tolower`.
 * After conversion, the strings are compared using standard lexicographical comparison operators.
 * The program prints -1 if the first string is lexicographically smaller than the second,
 * 1 if it is larger, and 0 if they are equal.
 *
 * Time Complexity: O(L), where L is the length of the strings. The `transform` operation
 * and the final string comparison both take linear time.
 * Space Complexity: O(L) to store the two input strings.
 */
int main() {
  string s1, s2;
  // Read the two strings to be compared.
  cin >> s1 >> s2;

  // Convert the first string to lowercase.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  // Convert the second string to lowercase.
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  /**
   * @brief Perform lexicographical comparison on the lowercase strings.
   * C++ string operators directly compare strings lexicographically.
   */
  if (s1 < s2)
    cout << "-1\n";
  else if (s1 == s2)
    cout << "0\n";
  else
    cout << "1\n";
  return 0;
}