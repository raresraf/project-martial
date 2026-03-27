/**
 * @file 281A.CPP
 * @brief This program capitalizes the first letter of a given input string if it is a lowercase alphabet.
 *
 * Algorithm:
 * 1. Read a string from standard input.
 * 2. Check if the string is not empty and its first character is a lowercase letter.
 * 3. If it is, convert the first character to uppercase by subtracting 32 (ASCII difference).
 * 4. Print the modified string to standard output.
 *
 * Time Complexity: O(L) where L is the length of the input string due to string length check and character access.
 * Space Complexity: O(L) to store the input string.
 */

#include <iostream>

using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Orchestrates the input, transformation (capitalization), and output of the string.
 * @return 0 on successful execution.
 */
int main() {
  string input;
  // Block Logic: Reads a single word (string) from standard input.
  cin >> input;
  /**
   * Block Logic: Checks if the input string is non-empty and if its first character
   * is a lowercase letter. If both conditions are met, it converts the first
   * character to its uppercase equivalent.
   *
   * Pre-condition: `input` string may or may not be empty.
   * Invariant: After this block, if `input` was not empty and its first character
   * was a lowercase letter, the first character will be uppercase. Otherwise,
   * the string remains unchanged.
   */
  if (input.length() > 0 && input[0] > 96) {
    // Inline: Converts a lowercase ASCII character to uppercase by subtracting 32.
    input[0] = input[0] - 32;
  }
  // Block Logic: Prints the potentially modified string to standard output, followed by a newline.
  cout << input << endl;
  return 0;
}