/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem.
 *
 * This program reads a single word, capitalizes its first letter, and then prints the modified word.
 * Algorithm: String manipulation with character case conversion.
 * Time Complexity: O(L) where L is the length of the input string, due to string input and output,
 *                  but the core logic for capitalization is O(1).
 * Space Complexity: O(L) to store the input string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to capitalize the first letter of an input word.
 *
 * Reads a word, converts its first character to uppercase, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  string s;
  // Block Logic: Read the input word from standard input.
  // Pre-condition: User provides a single word (string).
  // Invariant: 's' will contain the input word.
  cin >> s;

  // Block Logic: Capitalize the first character of the string.
  // The toupper() function converts a character to its uppercase equivalent.
  // Pre-condition: 's' is not empty.
  // Invariant: The first character of 's' will be uppercase.
  s[0] = toupper(s[0]);

  // Block Logic: Print the modified string to standard output.
  // Post-condition: The word with its first letter capitalized is displayed.
  cout << s;
}