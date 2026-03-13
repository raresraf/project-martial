/**
 * @file 281A.CPP
 * @brief Word Capitalization
 *
 * Capitalization is writing a word with its first letter as a capital letter.
 * Your task is to capitalize the given word.
 *
 * Note, that during capitalization all the letters except the first one
 * remains unchanged.
 *
 * @see https://codeforces.com/problemset/problem/281/A
 */
/* Accepted 60 ms 2144 KB */

#include <ctype.h>
#include <iostream>
#include <string>

int main() {
  std::string word;
  /**
   * @brief Reads the input word from standard input.
   */
  std::cin >> word;
  /**
   * @brief Capitalizes the first letter of the word.
   *
   * The `toupper` function is used to convert the character at the first
   * position of the string to its uppercase equivalent.
   */
  word.at(0) = toupper(word.at(0));
  /**
   * @brief Prints the capitalized word to standard output.
   */
  std::cout << word << '\n';
  return 0;
}
