/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces (Problem 281A).
 * @details This program reads a single word and capitalizes its first letter,
 * leaving the rest of the word unchanged.
 *
 * @section algorithm
 * Algorithm: Direct String Manipulation
 * 1. A custom `Capitalize` function is defined to handle the conversion of a
 *    character from lowercase to uppercase using ASCII arithmetic.
 * 2. The main function reads a word from standard input.
 * 3. It iterates through the word, character by character.
 * 4. For the first character (index 0), it calls `Capitalize` before printing.
 * 5. For all other characters, it prints them without modification.
 *
 * @section complexity
 * - Time Complexity: O(L), where L is the length of the word, due to the
 *   single pass through the string.
 * - Space Complexity: O(L) to store the input word.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Converts a character to its uppercase equivalent using ASCII arithmetic.
 * @param ch The character to convert.
 * @return The uppercase version of the character, or the original character
 *         if it is not a lowercase letter.
 * @note This implementation is ASCII-dependent and not a standard C++ practice.
 *       Using `toupper()` from `<cctype>` is generally preferred.
 */
int Capitalize(char ch) {
  // If `ch` is already uppercase (or not a letter), its ASCII value will be
  // less than 'a'. In this case, `ch - 'a'` is negative.
  if (ch - 'a' < 0)
    return ch;
  
  // Converts lowercase to uppercase by subtracting 32 (the ASCII offset).
  return ch - 32;
}

int main() {
  string word;
  cin >> word;

  /**
   * @brief Iterates through the word and prints the capitalized version.
   * A ternary operator is used to conditionally apply capitalization to only
   * the first letter.
   */
  for (size_t i = 0; i < word.size(); ++i) {
    cout << (char)((i > 0) ? word[i] : Capitalize(word[i]));
  }
  return 0;
}
