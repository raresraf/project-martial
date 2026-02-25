/**
 * @file 281A.CPP
 * @brief Solution for Codeforces Problem 281A - Word Capitalization.
 *
 * This program takes a single word as input and capitalizes its first letter,
 * leaving the rest of the word unchanged.
 *
 * Algorithm: Direct character modification. The program reads the input word
 * as a string, accesses its first character, converts it to uppercase using
 * `toupper()`, and then prints the modified string.
 * Time Complexity: O(L), where L is the length of the input word, primarily
 *                  due to string input/output. The capitalization itself is O(1).
 * Space Complexity: O(L) for storing the input string.
 */

#include <ctype.h>  // Required for the toupper() function.
#include <iostream> // Required for standard input/output operations (std::cin, std::cout).
#include <stdio.h>  // Although included, not explicitly used with stdio functions like printf/scanf.
#include <string>   // Required for std::string.

int main() {
  /**
   * @brief Main function to execute the Word Capitalization problem logic.
   *
   * Reads a word from standard input. It then capitalizes the first character
   * of this word and prints the resulting word to standard output.
   */
  // Declare a std::string variable to store the input word.
  std::string input;
  // Read the input word into the 'input' string.
  std::cin >> input;

  // Block Logic: Capitalize the first character.
  // Pre-condition: The 'input' string is not empty.
  // Access the first character of the string using index 0 and convert it to uppercase.
  // The result is assigned back to the same position in the string.
  input[0] = toupper(input[0]);

  // Output the modified string, followed by a newline.
  std::cout << input << std::endl;

  return 0; // Indicate successful program execution.
}
