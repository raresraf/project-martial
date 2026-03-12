/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem (Codeforces 96A).
 * @details The problem defines a "dangerous" situation in a football match as one
 *          where at least seven players of the same team are positioned consecutively.
 *          This program checks if a given sequence of players (represented by a
 *          string of '0's and '1's) constitutes a dangerous situation.
 *
 * @algorithm The algorithm checks for the existence of two specific substrings within
 *            the input string: "1111111" and "0000000". If either of these
 *            substrings is found, the situation is dangerous. The program implements
 *            a basic substring search function to perform this check.
 *
 * @time_complexity O(L), where L is the length of the input string. Although the naive
 *                  substring search can be O(L*M) in the worst case (where M is the
 *                  pattern length, 7 here), for a fixed pattern length it is effectively linear.
 * @space_complexity O(L) to store the input string.
 */
#include <bits/stdc++.h>
#include <string>
using namespace std;

/**
 * @brief A naive implementation to find if a string `s1` is a substring of `s2`.
 * @param s1 The substring to search for (the needle).
 * @param s2 The string to search in (the haystack).
 * @return The starting index of the first occurrence of `s1` in `s2`, or -1 if not found.
 */
int isSubstring(string s1, string s2) {
  int M = s1.length();
  int N = s2.length();
  // Iterate through the haystack string.
  for (int i = 0; i <= N - M; i++) {
    int j;
    // Check for pattern match starting at position i.
    for (j = 0; j < M; j++)
      if (s2[i + j] != s1[j])
        break;
    // If the inner loop completed, the pattern was found.
    if (j == M)
      return i;
  }
  return -1;
}

int main() {
  string s, s1 = "1111111", s0 = "0000000";
  getline(cin, s);
  
  // Check if a sequence of seven '1's exists.
  int res1 = isSubstring(s1, s);
  // Check if a sequence of seven '0's exists.
  int res0 = isSubstring(s0, s);
  
  /**
   * @brief A situation is not dangerous only if neither "1111111" nor "0000000" is found.
   * Otherwise, it is dangerous.
   */
  if (res1 == -1 && res0 == -1)
    cout << "NO";
  else
    cout << "YES";
  return 0;
}
