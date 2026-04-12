/**
 * @file 281A.CPP
 * @brief Logic for string capitalization/normalization.
 * 
 * Logic: Ensures that the first character of an input string is in its 
 * uppercase representation, leaving all subsequent characters in their original state.
 * 
 * Algorithm: Constant-time index-based state modification.
 * Time Complexity: $O(1)$ for the transformation (excluding I/O).
 * Space Complexity: $O(N)$ for the input string buffer.
 */

#include <ctype.h>
#include <iostream>
#include <string>

int main() {
  std::string word;
  
  // Functional Utility: Reads the input string from standard input.
  std::cin >> word;

  /**
   * Functional Utility: Normalizes the first character.
   * Logic: Uses toupper() to atomically transition the first character to 
   * uppercase if it is not already.
   */
  if (!word.empty()) {
    word.at(0) = (char)toupper(word.at(0));
  }
  
  // Output the final normalized string.
  std::cout << word << '\n';
  
  return 0;
}
