/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem (Codeforces 281A).
 *
 * @details This program reads a word and capitalizes its first letter.
 * If the first letter is already capitalized or is not a letter, it remains unchanged.
 *
 * @algorithm The program reads a single string from standard input. It then
 * targets the first character of the string (at index 0) and applies the
 * `toupper()` function to convert it to its uppercase equivalent. The modified
 * string is then printed to standard output.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the word. The complexity is
 * dominated by the string input/output operations.
 * Space Complexity: O(L), to store the input word.
 */
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <string>

int main() {
  std::string input;
  std::cin >> input;

  // Functional Logic: Capitalize the first character of the string.
  // The `toupper` function handles the conversion.
  input[0] = toupper(input[0]);

  std::cout << input << std::endl;

  return 0;
}
