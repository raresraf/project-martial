/**
 * @file 118A.CPP
 * @brief Solution for a string processing problem ("String Task").
 * @details The program processes a string based on a set of rules:
 *          1. It removes all vowel letters ('A', 'O', 'Y', 'E', 'U', 'I').
 *          2. Before each consonant, it inserts a '.' character.
 *          3. All uppercase consonants are replaced with their lowercase counterparts.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. For each character, check if it is a vowel (case-insensitive).
 *    a. If it is a vowel, do nothing (effectively deleting it from the output).
 *    b. If it is a consonant:
 *       i. Print a '.' character.
 *       ii. Convert the consonant to lowercase if it isn't already.
 *       iii. Print the lowercase consonant.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
using namespace std;
int main() {
  string input;
  char c;
  cin >> input;
  
  /**
   * @block
   * @description This loop iterates through the input string to process each character
   * according to the problem's rules.
   */
  for (int i = 0; i < input.length(); i++) {
    /**
     * @block
     * @description Checks if the character is a vowel (case-insensitively).
     * If it is, the loop continues to the next character, effectively deleting the vowel.
     */
    if (input[i] == 'a' || input[i] == 'A' || input[i] == 'e' ||
        input[i] == 'E' || input[i] == 'i' || input[i] == 'I' ||
        input[i] == 'o' || input[i] == 'O' || input[i] == 'u' ||
        input[i] == 'U' || input[i] == 'y' || input[i] == 'Y') {
      continue;
    } else {
      // If the character is a consonant, prefix it with a '.'.
      cout << ".";
      
      /**
       * @block
       * @description Converts the consonant to lowercase before printing.
       * It checks if the character's ASCII value is in the uppercase range.
       */
      if (input[i] >= 65 && input[i] <= 90) { // ASCII for 'A' to 'Z'
        // Convert to lowercase by adding 32 to the ASCII value.
        c = input[i] + 32;
        cout << c;
      } else {
        // Print the character as is if it's already lowercase.
        cout << input[i];
      }
    }
  }
}