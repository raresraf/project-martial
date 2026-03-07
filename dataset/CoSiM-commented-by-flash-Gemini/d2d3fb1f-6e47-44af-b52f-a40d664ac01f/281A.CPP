/**
 * @file 281A.CPP
 * @brief Solution to a competitive programming problem that capitalizes the first letter of a given word.
 * The program reads a single word, converts its first letter to uppercase, and prints the modified word.
 * Algorithm: String iteration and character manipulation.
 * Time Complexity: O(L), where L is the length of the input word.
 * Space Complexity: O(L) for storing the input word.
 */
#include <iostream> // For input/output operations.
#include <string>   // For string manipulation.

using namespace std;

/**
 * @brief Converts a given character to its uppercase equivalent if it's a lowercase letter.
 * If the character is already uppercase or not a letter, it returns the character unchanged.
 *
 * @param ch The character to convert.
 * @return int: The uppercase equivalent of the character, or the original character.
 * Note: Uses ASCII arithmetic; 'a' through 'z' are contiguous, and 'A' through 'Z' are contiguous,
 * with a fixed difference between corresponding lowercase and uppercase letters.
 */
int Capitalize(char ch) {
  // Conditional Logic: Checks if the character is a lowercase letter by comparing its ASCII value.
  // 'ch - 'a' < 0' means 'ch' has a smaller ASCII value than 'a', hence it's not a lowercase letter (or is 'a' itself).
  if (ch - 'a' < 0) // This check is essentially `ch < 'a'`, which means it's not a lowercase letter.
    return ch;      // Returns the character as is if it's not a lowercase letter.
  return ch - 32;   // Converts lowercase to uppercase by subtracting 32 (ASCII difference).
}

int main() {
  string word; // Declares 'word' to store the input string.
  // Reads the input word from standard input.
  cin >> word;
  
  /**
   * Block Logic: Iterates through the input word, capitalizing only the first character.
   * Invariant: At the start of each iteration, 'i' is the current character's index.
   * Pre-condition: 'word' contains a valid string.
   * Post-condition: The modified word (first letter capitalized) is printed.
   */
  for (size_t i = 0; i < word.size(); ++i) {
    // Conditional Logic: Prints the character. If it's the first character (i == 0),
    // it's capitalized using the `Capitalize` function. Otherwise, it's printed as is.
    cout << (char)((i > 0) ? word[i] : Capitalize(word[i]));
  }
  // No explicit newline is printed after the word, so subsequent output would be on the same line.
  return 0; // Indicates successful program execution.
}
