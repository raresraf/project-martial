/**
 * @file 112A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 112A "Petya and Strings".
 *
 * @details
 * This program performs a case-insensitive lexicographical comparison of two input
 * strings. It should print -1, 1, or 0 depending on whether the first string is
 * less than, greater than, or equal to the second.
 *
 * @algorithm
 * The algorithm first normalizes both strings to lowercase and then compares them.
 * 1. Read two strings from standard input.
 * 2. Manually iterate through each string, converting any uppercase characters to
 *    lowercase by adding 32 to their ASCII value.
 * 3. Use the `std::string::compare` method to compare the normalized strings.
 * 4. Print the result of the `compare` method directly.
 *
 * @note The C++ standard guarantees that `string::compare` returns <0, 0, or >0.
 * While many platforms return exactly -1 and 1, this is not guaranteed. However,
 * it is a common and accepted practice in competitive programming.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the length of the strings, for iterating
 *   and comparing them.
 * - Space Complexity: O(N) to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {

  string s1, s2;
  // Unused variables `c1`, `c2`.
  int c1 = 0, c2 = 0; 
  cin >> s1 >> s2;

  /**
   * @brief Manually convert the first string to lowercase.
   * It iterates through each character and checks if it's uppercase by comparing
   * its ASCII value to 'a'. If so, it adds 32 to convert it to lowercase.
   */
  for (char &i : s1) {
    if (i < 'a')
      i += 32;
  }

  /**
   * @brief Manually convert the second string to lowercase.
   */
  for (char &i : s2) {
    if (i < 'a')
      i += 32;
  }

  // Compare the normalized strings and print the result.
  cout << (s1.compare(s2)) << endl;

  return 0;
}
