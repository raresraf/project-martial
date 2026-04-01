/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem from Codeforces (112A).
 * @details This program compares two strings lexicographically, ignoring case. It outputs -1
 * if the first string is less than the second, 1 if the first is greater, and 0 if they are equal.
 *
 * Algorithm: Lexicographical Comparison.
 * 1. Read two strings from standard input.
 * 2. Iterate through the strings character by character.
 * 3. In each iteration, convert the characters at the current position to lowercase.
 * 4. Compare the lowercase characters.
 * 5. If they are different, determine which string is lexicographically smaller and stop.
 * 6. If the loop completes without finding any differences, the strings are equal.
 *
 * Time Complexity: O(L) where L is the length of the shorter string, as the loop terminates
 * at the first differing character or at the end of the strings.
 * Space Complexity: O(L) to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams for faster I/O.
   */
  ios::sync_with_stdio(false);
  string s1, s2;
  cin >> s1;
  cin >> s2;
  int flag = 0;

  /**
   * Block Logic: Performs a case-insensitive lexicographical comparison.
   * Invariant: At the beginning of each iteration `i`, the prefixes s1[0...i-1] and
   * s2[0...i-1] are lexicographically equal when ignoring case.
   */
  for (int i = 0; i < s1.length(); ++i) {
    if (tolower(s1[i]) < tolower(s2[i])) {
      flag = -1;
      break;
    }
    if (tolower(s1[i]) > tolower(s2[i])) {
      flag = 1;
      break;
    }
  }
  // Output the result: -1, 1, or 0.
  cout << flag << "\n";

  return 0;
}