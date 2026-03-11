/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem.
 *
 * This program reads a single word (or line), capitalizes its first letter, and then prints the modified word.
 * Algorithm: String manipulation with manual ASCII-based character case conversion.
 * Time Complexity: O(L) where L is the length of the input string, due to string input and output,
 *                  but the core logic for capitalization is O(1).
 * Space Complexity: O(L) to store the input string.
 */

#include <iostream>

using namespace std;

/**
 * @brief Main function to capitalize the first letter of an input word.
 *
 * Reads a word, converts its first character to uppercase, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  string s = "";

  // Block Logic: Read the input word (or line) from standard input.
  // Using getline allows reading strings with spaces.
  // Pre-condition: User provides a string.
  // Invariant: 's' will contain the input string.
  getline(cin, s);

  // Block Logic: Capitalize the first character of the string if it is a lowercase letter.
  // Checks if the first character's ASCII value is within the lowercase range (97-122).
  // If true, subtracts 32 to convert it to its uppercase ASCII equivalent.
  // Pre-condition: 's' is not empty.
  // Invariant: The first character of 's' will be uppercase if it was a lowercase letter.
  if (s[0] >= 97 && s[0] <= 122) {
    s[0] = s[0] - 32;
  }

  // Block Logic: Print the modified string to standard output.
  // Post-condition: The string with its first letter capitalized is displayed.
  cout << s;

  return 0;
}
