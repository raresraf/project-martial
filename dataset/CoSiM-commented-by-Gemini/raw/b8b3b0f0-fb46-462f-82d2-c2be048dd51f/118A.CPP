/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem (Codeforces 118A).
 *
 * @details
 * This program reformats a string according to three rules:
 * 1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I', and their lowercase versions).
 * 2. Insert a '.' before each consonant.
 * 3. Convert all consonants to lowercase.
 *
 * This implementation uses a multi-pass approach with C-style strings:
 * - **Pass 1 (Vowel Removal):** It iterates through the input string `s` and copies
 *   only the consonants into a temporary buffer `temp`.
 * - **Pass 2 (Case Conversion):** It iterates through the `temp` buffer (now containing
 *   only consonants) and converts all characters to lowercase.
 * - **Pass 3 (Output Formatting):** It iterates through the modified `temp` buffer
 *   and prints each character, prepended with a '.'.
 *
 * While functionally correct, this approach is less efficient than a single-pass
 * solution due to the use of an intermediate buffer and multiple iterations.
 *
 * Algorithm: Three-Pass String Transformation
 * Time Complexity: O(L), where L is the length of the string, as each pass is linear.
 * Space Complexity: O(L), due to the storage of the input string `s` and the
 *                   intermediate buffer `temp`.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[101];
  char temp[101];
  int j = 0;
  cin >> s;

  /**
   * @brief Pass 1: Filter out vowels.
   * @details This loop iterates through the input string `s`. If a character is
   * not a vowel, it is appended to the `temp` string.
   */
  for (int i = 0; s[i] != '\0'; i++) {
    if (s[i] != 'A' && s[i] != 'a' && s[i] != 'E' && s[i] != 'e' &&
        s[i] != 'I' && s[i] != 'i' && s[i] != 'O' && s[i] != 'o' &&
        s[i] != 'U' && s[i] != 'u' && s[i] != 'y' && s[i] != 'Y') {
      temp[j] = s[i];
      j++;
    }
  }
  temp[j] = '\0'; // Null-terminate the temporary string.

  /**
   * @brief Pass 2: Convert all characters in the temporary string to lowercase.
   * @details This ensures all consonants are in their lowercase form before printing.
   * The code adds 32 to the ASCII value, which is a common way to convert
   * from uppercase to lowercase for English alphabet characters.
   */
  for (int i = 0; temp[i] != '\0'; i++) {
    if (temp[i] >= 'A' && temp[i] <= 'Z') {
      temp[i] = temp[i] + 32;
    }
  }

  /**
   * @brief Pass 3: Print the final formatted string.
   * @details This loop iterates through the fully processed `temp` string and prints
   * each character with a preceding '.'
   */
  for (int i = 0; temp[i] != '\0'; i++) {
    cout << "." << temp[i];
  }
}