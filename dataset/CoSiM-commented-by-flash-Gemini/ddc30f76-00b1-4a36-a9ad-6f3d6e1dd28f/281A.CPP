/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem.
 *
 * This program reads a single word, capitalizes its first letter, and then prints the modified word.
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
  char c[1000];
  // Block Logic: Read the input word from standard input into a C-style char array.
  // Pre-condition: User provides a single word (string) whose length does not exceed 999 characters.
  // Invariant: 'c' will contain the input word.
  cin >> c;

  // Block Logic: Manually capitalize the first character of the string if it's a lowercase letter.
  // ASCII values: 'a' is 97, 'A' is 65. Subtracting 32 converts lowercase to uppercase.
  // Pre-condition: 'c' is not empty.
  // Invariant: The first character of 'c' will be uppercase if it was a lowercase letter.
  if (c[0] >= 97)
    c[0] -= 32;

  // Block Logic: Print the modified string to standard output.
  // Post-condition: The word with its first letter capitalized is displayed, followed by a newline.
  cout << c << endl;

  return 0;
}
