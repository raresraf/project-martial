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

#include <bits/stdc++.h> // Includes various standard libraries for competitive programming.
using namespace std;       // Uses the standard namespace for convenience.

/**
 * @brief Main function to capitalize the first letter of an input word.
 *
 * Reads a word, converts its first character to uppercase, and prints the result.
 * @return 0 on successful execution.
 */
int main() {
  char w[1000]; // Declare a character array to store the input word. Max length 999 + null terminator.

  // Block Logic: Read the input word from standard input into a C-style char array.
  // Pre-condition: User provides a single word (string) whose length does not exceed 999 characters.
  // Invariant: 'w' will contain the input word.
  cin >> w;

  int a = w[0]; // Store the ASCII value of the first character for comparison. (Note: 'a' is unused beyond this).

  // Block Logic: Manually capitalize the first character of the string if it's a lowercase letter.
  // Checks if the first character's ASCII value is within the lowercase range (>= 'a').
  // If true, subtracts 32 to convert it to its uppercase ASCII equivalent ('A' - 'a' is -32).
  // Pre-condition: 'w' is not empty.
  // Invariant: The first character of 'w' will be uppercase if it was a lowercase letter.
  if (w[0] >= 'a') {
    w[0] = a - 32;
  }

  // Block Logic: Print the modified string to standard output.
  // Post-condition: The word with its first letter capitalized is displayed, followed by a newline.
  cout << w << endl;

  return 0; // Indicate successful execution.
}
