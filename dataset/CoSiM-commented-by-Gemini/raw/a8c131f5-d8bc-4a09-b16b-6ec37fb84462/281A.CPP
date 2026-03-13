/**
 * @file 281A.CPP
 * @brief Solution for CodeForces Problem 281A - "Word Capitalization".
 * 
 * @details
 * This program reads a single word and capitalizes its first letter. 
 * All other letters remain unchanged.
 *
 * @algorithm String manipulation
 * 1. Read the input word into a `std::string`.
 * 2. Use the `toupper()` function from `<cctype>` to convert the first character
 *    (at index 0) to its uppercase equivalent. `toupper()` correctly handles
 *    characters that are already uppercase or not letters.
 * 3. Print the modified string.
 *
 * @time_complexity O(1), as it only accesses and modifies the first character.
 * @space_complexity O(L), where L is the length of the input word.
 *
 * @note The integer variables `ln`, `i`, and `j` are declared but never used.
 */
#include <iostream>
#include <string>
#include <cctype> // For toupper()

using namespace std;

int main() {
  // Unused variables.
  // int ln, i, j; 
  
  string str;
  cin >> str;

  // Ensure the string is not empty before accessing the first character.
  if (!str.empty()) {
    /**
     * @brief Capitalize the first letter using toupper().
     * This is the standard and portable way to convert a character to uppercase.
     */
    str[0] = toupper(str[0]);
  }

  cout << str;

  return 0;
}
