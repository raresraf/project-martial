/**
 * @file 281A.CPP
 * @brief Solution for a "Word Capitalization" problem.
 *
 * Algorithm:
 * This program reads a word and capitalizes its first letter if it is currently in lowercase.
 * It achieves this by directly manipulating the ASCII value of the first character.
 * It checks if the character's ASCII value is in the lowercase range and, if so, converts
 * it to uppercase by subtracting 32.
 *
 * Time Complexity: O(N), where N is the length of the word, dominated by the string input/output operations.
 * The capitalization logic itself is O(1).
 *
 * Space Complexity: O(N), to store the input word.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // The input word.
  string word;

  // Read the entire line as the word.
  getline(cin, word);
  
  // Create an iterator pointing to the first character of the word.
  string::iterator it = word.begin();
  
  /**
   * @brief This block capitalizes the first letter if it's lowercase.
   * It uses ASCII value manipulation for the check and conversion.
   * Pre-condition: 'it' points to the first character of a non-empty string.
   */
  // Check if the ASCII value of the first character is greater than 90 ('Z').
  // This is a check for a lowercase letter.
  if (*it > 90)
    // If it's a lowercase letter, subtract 32 to convert it to uppercase.
    *it -= 32;
  
  // Print the modified word.
  cout << word << endl;

  return 0;
}
