// link : https://codeforces.com/contest/112/problem/A

/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces.
 * @details The problem is to compare two strings lexicographically, ignoring
 * case differences. The output should be -1, 1, or 0.
 *
 * @algorithm
 * The code performs a case-insensitive comparison character by character.
 * 1. Read two strings.
 * 2. Loop through the strings from the first to the last character.
 * 3. In each iteration:
 *    a. Convert the current characters from both strings to lowercase. This
 *       is done using ASCII arithmetic (`char += ' '`) which adds 32.
 *    b. Compare the two lowercase characters.
 *    c. If they are different, the lexicographical order is determined. Print
 *       -1 or 1 and terminate the program immediately.
 * 4. If the loop completes without any differences being found, the strings
 *    are equal. Print 0.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings. The loop
 * terminates at the first differing character.
 * Space Complexity: O(L) to store the two strings.
 */
#include <bits/stdc++.h>

using namespace std;

string s1, s2; // Input strings, declared globally.
int k;         // Global integer, declared but unused.

main() {

  cin >> s1 >> s2;

  /**
   * @block
   * @brief Compare strings character by character in a case-insensitive manner.
   */
  for (int i = 0; i < s1.size(); i++) {

    // Convert current characters to lowercase using ASCII arithmetic.
    // This assumes that any character with an ASCII value less than 'a'
    // is an uppercase letter that needs conversion.
    if (s1[i] < 'a')
      s1[i] += ' '; // Adding ' ' (ASCII 32) converts 'A'-'Z' to 'a'-'z'.

    if (s2[i] < 'a')
      s2[i] += ' ';

    // If a difference is found, print the result and exit.
    if (s1[i] < s2[i]) {
      cout << "-1";
      return 0;
    }
    if (s1[i] > s2[i]) {
      cout << "1";
      return 0;
    }
  }

  // If the loop completes, the strings are equal.
  cout << "0";
}
