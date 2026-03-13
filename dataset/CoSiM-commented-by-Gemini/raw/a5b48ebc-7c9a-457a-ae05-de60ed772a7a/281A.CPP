/**
 * @file 281A.CPP
 * @brief Solution for CodeForces Problem 281A - "Word Capitalization".
 * 
 * @details
 * This program reads a single word and capitalizes its first letter. 
 * All other letters remain unchanged.
 *
 * @algorithm String manipulation
 * @time_complexity O(1), as it only modifies the first character of the string.
 * @space_complexity O(L), where L is the length of the input word.
 *
 * @note The check `slowo[0] > 90` is a way to test if the character is lowercase
 * in the ASCII table (ASCII 'Z' is 90). A more portable method would be `islower()`.
 * The subtraction by 32 is a common ASCII trick to convert a lowercase letter
 * to its uppercase equivalent. A more portable method is `toupper()`.
 */
#include <cctype>
#include <iostream>
#include <string>

int main() {
  std::string slowo; // "slowo" means "word" in Polish.
  std::cin >> slowo;

  /**
   * @brief Capitalize the first letter.
   * Pre-condition: `slowo` contains the input word.
   * Post-condition: The first character of `slowo` is uppercase.
   */
  // Checks if the first character is a lowercase letter (ASCII value > 90).
  if (slowo[0] >= 'a' && slowo[0] <= 'z') {
    // Converts the first letter to uppercase by subtracting the ASCII offset.
    slowo[0] = toupper(slowo[0]);
  }
  
  // Output the modified word.
  std::cout << slowo;

  return 0;
}
