/**
 * @file 112A.CPP
 * @brief Solution for Codeforces problem 112A, "Petya and Strings".
 *
 * @description
 * This program performs a case-insensitive lexicographical comparison of two
 * input strings. It outputs -1 if the first string is lexicographically smaller
 * than the second, 1 if it is larger, and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings, s1 and s2.
 * 2. Initialize a result variable to 0 (the default for equality).
 * 3. Iterate through both strings from left to right, character by character.
 * 4. In each iteration, convert the current characters from both strings to
 *    uppercase to ensure case-insensitivity.
 * 5. Compare the uppercase characters.
 *    - If the character from s1 is greater than the one from s2, set result to 1
 *      and terminate the loop.
 *    - If the character from s1 is less than the one from s2, set result to -1
 *      and terminate the loop.
 *    - If the characters are equal, continue to the next pair.
 * 6. Print the final result.
 *
 * @time_complexity O(L), where L is the length of the strings. The comparison
 *   stops at the first differing character.
 * @space_complexity O(L) to store the input strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s1, s2;
  int cmp1, cmp2;
  // 'result' stores the outcome of the comparison: -1, 0, or 1.
  int result = 0;
  cin >> s1;
  cin >> s2;

  // Block Logic: Iterate through the strings, comparing them character by character.
  for (int i = 0; i < s1.size(); i++) {
    // Convert the character from s1 to uppercase for case-insensitive comparison.
    // 97 is the ASCII value for 'a'; subtracting 32 converts to uppercase.
    if (s1[i] >= 97) {
      s1[i] -= 32;
    }
    // Convert the character from s2 to uppercase.
    if (s2[i] >= 97) {
      s2[i] -= 32;
    }

    // Perform lexicographical comparison on the uppercase characters.
    if (s1[i] > s2[i]) {
      result = 1;
      break; // First string is greater, no need to check further.
    }
    if (s1[i] < s2[i]) {
      result = -1;
      break; // First string is smaller, no need to check further.
    }
  }
  cout << result;
  return 0;
}
