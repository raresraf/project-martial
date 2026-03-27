/**
 * @file 281A.CPP
 * @brief This program capitalizes the first letter of a given input string if it is a lowercase alphabet.
 *
 * Algorithm:
 * 1. Read a single word (string) from standard input.
 * 2. Check if the first character of the string falls within the ASCII range for lowercase letters (97 to 122).
 * 3. If it is a lowercase letter, convert the first character to uppercase using the `toupper` function.
 * 4. Print the modified string to standard output.
 *
 * Time Complexity: O(L) where L is the length of the input string due to string length check and character access.
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input, transformation (capitalization), and output of the string.
 * @return 0 on successful execution.
 */
int main() {
  string s;
  // Block Logic: Reads a single word (string) from standard input.
  cin >> s;
  /**
   * Block Logic: Checks if the first character of the string is a lowercase letter based on its ASCII value.
   * If it is, the character is converted to its uppercase equivalent.
   *
   * Pre-condition: The string 's' has been populated.
   * Invariant: If the first character is a lowercase letter, it will be transformed to uppercase; otherwise, it remains unchanged.
   */
  if (s[0] < 123 and s[0] > 96) {
    // Inline: Converts the character to its uppercase equivalent.
    s[0] = toupper(s[0]);
  }
  // Block Logic: Prints the potentially modified string to standard output, followed by a newline.
  cout << s << endl;
  return 0;
}