/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem.
 *
 * @details
 * The program takes a single word as input and capitalizes its first letter.
 * If the first letter is already capitalized or is not a letter, the word
 * remains unchanged.
 *
 * Algorithm:
 * 1. Read the input word into a string.
 * 2. Check if the first character of the string is a lowercase letter.
 *    The code does this by checking if its ASCII value is within the range
 *    of 'a' to 'z' (97 to 122).
 * 3. If it is a lowercase letter, convert it to its uppercase equivalent
 *    in-place.
 * 4. Print the resulting string.
 *
 * Time Complexity: O(L) where L is the length of the string, due to I/O. The capitalization logic itself is O(1).
 * Space Complexity: O(L) to store the string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  /**
   * @brief Checks if the first character is a lowercase letter.
   * This is done by comparing its ASCII value against the range for 'a' through 'z'.
   * ASCII 'a' is 97, 'z' is 122. The condition `s[0] > 96 && s[0] < 123`
   * correctly identifies lowercase English letters.
   */
  if (s[0] < 123 and s[0] > 96) {
    // If it's a lowercase letter, convert it to uppercase.
    s[0] = toupper(s[0]);
  }
  cout << s << endl;
  return 0;
}