/**
 * @file 112A.CPP
 * @brief Solution to Codeforces problem 112A - Petya and Strings.
 * @description This program performs a case-insensitive lexicographical comparison of two input strings.
 * It outputs -1 if the first string is lexicographically smaller than the second, 1 if it is larger,
 * and 0 if they are equal.
 *
 * Algorithm: The program first reads two strings. It then converts both strings entirely to
 * lowercase to ensure the comparison is case-insensitive. Finally, it uses the standard
 * string comparison operators to determine the lexicographical order and prints the
 * corresponding result (-1, 1, or 0).
 *
 * Time Complexity: O(N), where N is the length of the strings, due to the loop for converting
 * to lowercase and the subsequent comparison.
 * Space Complexity: O(N) to store the two input strings.
 */
#include <algorithm>
#include <iostream>
using namespace std;
string a, b;
int main() {
  cin >> a >> b;

  /**
   * Block Logic: Convert both strings to lowercase for case-insensitive comparison.
   * ASCII values for uppercase letters are smaller than for lowercase letters.
   * The check `a[i] < 92` is a way to identify uppercase English letters, and adding 32 converts them to lowercase.
   */
  for (int i = 0; i < a.size(); i++) {
    if (a[i] < 92) { // Checks if character is uppercase
      a[i] += 32;    // Converts to lowercase
    }
    if (b[i] < 92) { // Checks if character is uppercase
      b[i] += 32;    // Converts to lowercase
    }
  }

  /**
   * Block Logic: Perform lexicographical comparison on the normalized (lowercase) strings.
   */
  if (a < b) {
    cout << -1;
  } else if (a > b) {
    cout << 1;
  } else if (a == b) {
    cout << 0;
  }
}
