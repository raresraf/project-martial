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

// 281A - Word Capitalization // Problem ID and Name

#include <iostream> // For standard I/O operations (cin, cout).

using namespace std; // Use the standard namespace for convenience.

/**
 * @brief Main function to capitalize the first letter of an input word.
 *
 * Reads a word, converts its first character to uppercase, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  string word; // Declare a string variable to store the input word.

  // Block Logic: Read the input word from standard input.
  // Pre-condition: User provides a single word (string).
  // Invariant: 'word' will contain the input word.
  cin >> word;

  // Block Logic: Capitalize the first character of the string.
  // The toupper() function converts a character to its uppercase equivalent.
  // The result is cast back to char and assigned to the first character of the string.
  // Pre-condition: 'word' is not empty.
  // Invariant: The first character of 'word' will be uppercase.
  word[0] = char(toupper(word[0]));

  // Block Logic: Print the modified string to standard output.
  // Post-condition: The word with its first letter capitalized is displayed.
  cout << word;

  return 0; // Indicate successful execution.
}
