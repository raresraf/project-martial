/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of an input string.
 *
 * This program reads a single word from standard input and ensures that its
 * first letter is capitalized. It then prints the modified word to standard
 * output.
 *
 * Algorithm:
 * 1. Read a word from standard input into a std::string.
 * 2. Access the first character of the string using index 0.
 * 3. Convert this character to its uppercase equivalent using `toupper()`.
 * 4. The string is modified in-place.
 * 5. Print the entire modified string.
 *
 * Time Complexity: O(L), where L is the length of the input string. This is
 * primarily due to the time taken for reading and writing the string. The
 * capitalization operation itself is O(1).
 * Space Complexity: O(L) to store the input string.
 */
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <string>

int main() {
  std::string input;
  // Read a single word from standard input.
  std::cin >> input;

  /**
   * @brief Capitalize the first character of the string.
   * This is an in-place modification. The `toupper` function handles
   * characters that are already uppercase or not letters.
   */
  input[0] = toupper(input[0]);

  // Print the resulting string to standard output.
  std::cout << input << std::endl;

  return 0;
}
