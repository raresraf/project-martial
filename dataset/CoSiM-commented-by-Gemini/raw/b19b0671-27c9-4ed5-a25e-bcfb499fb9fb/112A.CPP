/**
 * @file 112A.CPP
 * @brief Solution for the "Petya and Strings" problem.
 * @details This program compares two strings lexicographically, ignoring differences in case.
 * It outputs -1 if the first string is lexicographically smaller, 1 if it is larger,
 * and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings, `s1` and `s2`.
 * 2. Iterate through both strings, converting each character to its lowercase equivalent
 *    using the `tolower()` function. This makes the subsequent comparison case-insensitive.
 * 3. Use the standard overloaded comparison operators for `std::string` (`==`, `>`) to
 *    determine the lexicographical relationship between the two modified strings.
 * 4. Print the result (-1, 1, or 0) based on the comparison outcome.
 *
 * @complexity
 * - Time: O(L), where L is the length of the strings. This is for the combination of reading,
 *   converting to lowercase, and the final comparison.
 * - Space: O(L) to store the two input strings.
 */
#include <cctype>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  string s1, s2; // Input strings to be compared.
  
  // The loop will execute once for the given problem constraints.
  while (cin >> s1 >> s2) {
    /**
     * @brief This loop converts both strings to lowercase for case-insensitive comparison.
     * The loop condition `i < s1.size(), j < s2.size()` is valid but could be simplified
     * to `i < s1.size()` since the strings are guaranteed to be of equal length.
     */
    for (int i = 0, j = 0; i < s1.size(), j < s2.size(); i++, j++) {
      s1[i] = tolower(s1[i]);
      s2[j] = tolower(s2[j]);
    }
    
    /**
     * @brief This block performs the lexicographical comparison and prints the result.
     * `std::string` overloads the comparison operators to perform lexicographical checks.
     */
    if (s1 == s2) {
      cout << "0" << endl;
    } else if (s1 > s2) {
      cout << "1" << endl;
    } else {
      cout << "-1" << endl;
    }
  }
}