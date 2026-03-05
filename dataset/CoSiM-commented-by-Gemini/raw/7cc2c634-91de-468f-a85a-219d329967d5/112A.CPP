/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A, "Petya and Strings".
 * @details The program performs a case-insensitive lexicographical comparison
 * of two input strings.
 *
 * Algorithm:
 * 1. Read two strings, `a` and `b`.
 * 2. Iterate through both strings and convert all uppercase characters to
 *    lowercase. This ensures the subsequent comparison is case-insensitive.
 * 3. Use the standard C++ string comparison operators (`<`, `>`, `==`) to
 *    compare the two now-lowercase strings lexicographically.
 * 4. Print -1 if the first string is lexicographically smaller than the second.
 * 5. Print 1 if the first string is lexicographically larger than the second.
 * 6. Print 0 if the strings are equal.
 *
 * Time Complexity: O(L), where L is the length of the strings, due to the
 * iteration for case conversion and the comparison.
 * Space Complexity: O(L) to store the two strings.
 */
#include <algorithm>
#include <iostream>
using namespace std;
string a, b;
int main() {
  cin >> a >> b;
  /**
   * @brief This loop iterates through the strings to convert them to lowercase.
   * The condition `a[i] < 92` is a way to check for uppercase letters in ASCII,
   * as 'Z' is 90. Adding 32 converts an uppercase character to its lowercase
   * equivalent (e.g., 'A' (65) + 32 = 'a' (97)).
   */
  for (int i = 0; i < a.size(); i++) {
    if (a[i] < 92) {
      a[i] += 32;
    }
    if (b[i] < 92) {
      b[i] += 32;
    }
  }
  /**
   * @brief Performs a lexicographical comparison on the lowercase strings and
   * prints the result according to the problem statement.
   */
  if (a < b) {
    cout << -1;
  } else if (a > b) {
    cout << 1;
  } else if (a == b) {
    cout << 0;
  }
}