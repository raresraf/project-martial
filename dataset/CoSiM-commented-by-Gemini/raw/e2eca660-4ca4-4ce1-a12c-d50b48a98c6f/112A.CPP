/**
 * @file 112A.CPP
 * @brief Solution to Codeforces problem 112A - Petya and Strings.
 * @description This program performs a case-insensitive lexicographical comparison of two
 *              input strings and prints -1, 1, or 0 depending on whether the first
 *              string is less than, greater than, or equal to the second.
 *
 * Algorithm: The program iterates through both strings simultaneously, character by
 *            character. In each step, it converts the characters from both strings at
 *            the current position to lowercase and compares them. If a difference is
 *            found, it immediately prints the result (1 or -1) and terminates. If the
 *            loop completes without finding any differences, the strings are considered
 *            equal, and 0 is printed. This approach is efficient as it stops at the
 *            first point of difference.
 *
 * Time Complexity: O(K), where K is the index of the first differing character. In the
 *                  worst case (equal strings), the complexity is O(N), where N is the
 *                  length of the strings.
 * Space Complexity: O(N) to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s1, s2;
  cin >> s1 >> s2;
  int i;
  /**
   * Block Logic: Iterate through the strings, comparing them character by character
   * after converting to lowercase.
   */
  for (i = 0; i < s1.length(); i++) {
    // Convert characters to lowercase for case-insensitive comparison.
    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);

    // If a character in the first string is greater, the first string is lexicographically larger.
    if (s1[i] > s2[i]) {
      cout << "1" << endl;
      return 0; // Exit immediately as the result is determined.
    }
    // If a character in the first string is smaller, the first string is lexicographically smaller.
    if (s1[i] < s2[i]) {
      cout << "-1" << endl;
      return 0; // Exit immediately.
    }
  }

  // If the loop completes, no differences were found, so the strings are equal.
  cout << "0" << endl;

  return 0;
}
