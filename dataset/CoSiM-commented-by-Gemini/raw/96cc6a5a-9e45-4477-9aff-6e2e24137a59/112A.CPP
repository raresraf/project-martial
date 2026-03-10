/**
 * @file 112A.CPP
 * @brief Solution for a competitive programming problem involving case-insensitive string comparison.
 * This program compares two strings lexicographically, ignoring case, and outputs
 * -1 if the first string is less than the second,
 *  1 if the first string is greater than the second,
 *  0 if they are equal.
 *
 * @algorithm Lexicographical comparison after converting both strings to lowercase.
 * @time_complexity O(N), where N is the length of the strings, due to linear scans for conversion and comparison.
 * @space_complexity O(N) for storing the two input strings.
 */
#include <bits/stdc++.h>

using namespace std;

#define ll long long

/**
 * @brief Converts an uppercase character to its lowercase equivalent.
 * @param s The character to convert.
 * @return The lowercase version of the character. If the character is already
 *         lowercase or not a letter, it is returned unchanged.
 */
char down(char s) { return (s <= 'Z' ? s + 32 : s); }

int main() {
  string s1, s2;
  cin >> s1 >> s2;

  /**
   * @block
   * @brief Normalizes both strings to lowercase.
   * This ensures that the subsequent comparison is case-insensitive.
   */
  for (int i = 0; i < s1.length(); i++) {
    s1[i] = down(s1[i]);
  }
  for (int i = 0; i < s1.length(); i++) {
    s2[i] = down(s2[i]);
  }

  /**
   * @block
   * @brief Performs lexicographical comparison of the normalized strings.
   * It iterates through the strings and compares them character by character.
   * The loop terminates and prints the result as soon as a difference is found.
   */
  for (int i = 0; i < s1.length(); i++) {
    if (s1[i] > s2[i]) {
      cout << 1 << "\n";
      return 0;
    } else if (s1[i] < s2[i]) {
      cout << -1 << "\n";
      return 0;
    }
  }

  // If the loop completes without finding any differences, the strings are equal.
  cout << 0 << "\n";
  return 0;
}
