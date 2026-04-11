/**
 * @file 281A.CPP
 * @brief This program takes a single word as input, capitalizes its first letter,
 * and then prints the modified word.
 *
 * Algorithm: String manipulation (character modification).
 * Time Complexity: O(L) where L is the length of the input word.
 * Space Complexity: O(L) to store the input word.
 */

#include <iostream>
#include <string>
#include <cctype> // Required for toupper

/**
 * @brief Prints the given string 'v' to standard output followed by a newline.
 * @param v The string to be printed.
 */
void answer(const std::string &v) { std::cout << v << '
'; }

/**
 * @brief Capitalizes the first letter of the input 'word' if it's not empty,
 * and then calls the 'answer' function to print the modified word.
 * @param word The string whose first letter needs to be capitalized.
 */
void solve(std::string word) {
  /**
   * Block Logic: Checks if the word is not empty before attempting to capitalize the first character.
   * Pre-condition: 'word' is a std::string.
   * Invariant: If 'word' is not empty, its first character will be converted to uppercase.
   */
  if (!word.empty())
    // Functional Utility: Converts the first character of the word to uppercase.
    word[0] = toupper(word[0]);

  // Block Logic: Outputs the (potentially modified) word.
  answer(word);
}

int main() {
  // Functional Utility: Declares a string 'word' to store the input.
  std::string word;
  // Block Logic: Reads a single word from standard input.
  std::cin >> word;

  // Block Logic: Calls the 'solve' function to process and print the word.
  solve(word);

  return 0;
}
