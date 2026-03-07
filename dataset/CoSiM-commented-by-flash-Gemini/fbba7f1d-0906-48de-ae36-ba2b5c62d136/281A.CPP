/*
  Contest 281
  Problem A : Word capitalization
  Author : Rakesh Kumar
  Date: 18/01/2020
*/

/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" competitive programming problem (Codeforces 281A).
 * The program reads a single word, ensures its first letter is capitalized, and then prints the word.
 * If the first letter is already uppercase, it remains unchanged. If it's lowercase, it's converted to uppercase.
 * Algorithm: String manipulation and character case conversion.
 * Time Complexity: O(L), where L is the length of the input word (due to getline and printf, and potential string copy).
 * Space Complexity: O(L) for storing the input word.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, including string, iostream, cctype.

int main() {
  std::string str; // Declares 'str' to store the input word.
  // Reads an entire line (word) from standard input into 'str'.
  std::getline(std::cin, str);
  
  /**
   * Conditional Logic: Checks if the first character of the string is a lowercase letter.
   * `std::islower()` is a standard library function from `<cctype>` (included via `<bits/stdc++.h>`).
   */
  if (std::islower(str[0]))
    // If the first character is lowercase, converts it to uppercase by subtracting 32 (ASCII difference).
    str[0] -= 32;
  
  // Prints the modified string to standard output, followed by a newline.
  // `printf` is used here with `str.c_str()` to get a C-style string representation.
  printf("%s\n", str.c_str());

  return 0; // Indicates successful program execution.
}
