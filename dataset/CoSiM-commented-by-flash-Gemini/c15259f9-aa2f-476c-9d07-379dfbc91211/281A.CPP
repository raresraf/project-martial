/**
 * @file 281A.CPP
 * @brief This program reads a line of text, capitalizes its first letter if it is a lowercase alphabet, and prints the modified string.
 *
 * Algorithm:
 * 1. Read an entire line of text from standard input into a string.
 * 2. Obtain an iterator to the beginning of the string.
 * 3. Check if the first character (pointed to by the iterator) has an ASCII value greater than 90 (which means it's a lowercase letter if it's an alphabet).
 * 4. If it is a lowercase letter, convert it to uppercase by subtracting 32 from its ASCII value.
 * 5. Print the modified string to standard output.
 *
 * Time Complexity: O(L) where L is the length of the input string, mainly due to reading the line and iterating to the first character.
 * Space Complexity: O(L) to store the input string.
 */

#include <iostream>

using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Manages the program flow from reading input to processing and displaying the output.
 * @return 0 on successful execution.
 */
int main() {
  string word;

  // Block Logic: Reads an entire line from standard input, including spaces, until a newline character is encountered.
  getline(cin, word);
  // Block Logic: Initializes an iterator to point to the beginning of the string.
  string::iterator it = word.begin();
  /**
   * Block Logic: Checks if the first character of the string is a lowercase letter (ASCII value > 90 implies it could be 'a' through 'z').
   * If true, it converts the character to its uppercase equivalent.
   *
   * Pre-condition: The string 'word' has been populated.
   * Invariant: If the first character is a lowercase letter, it will be transformed to uppercase; otherwise, it remains unchanged.
   */
  if (*it > 90)
    // Inline: Converts a lowercase ASCII character to uppercase by subtracting 32.
    *it -= 32;
  // Block Logic: Prints the potentially modified string to standard output, followed by a newline.
  cout << word << endl;

  return 0;
}