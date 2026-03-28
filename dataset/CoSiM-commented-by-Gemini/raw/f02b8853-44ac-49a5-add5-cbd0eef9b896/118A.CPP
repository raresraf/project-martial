/**
 * @file 118A_stringtask.cpp
 * @brief Processes a string by removing vowels, adding a dot before
 *        consonants, and converting them to lowercase.
 *
 * This program takes a string as input, removes all vowels (A, O, Y, E, U, I),
 * prefixes each consonant with a '.', and converts any uppercase consonants to
 * their lowercase equivalents.
 *
 * @b Algorithm: The program iterates through the input string character by
 *    character. It checks if the character is a vowel (case-insensitively).
 *    If it is, the character is skipped. If it's a consonant, a '.' is printed,
 *    followed by the lowercase version of the consonant.
 *
 * @b Time_Complexity: O(L), where L is the length of the input string, due to
 *    the single pass through the string.
 * @b Space_Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string.h>
#include <string>

using namespace std;

int main() {
  string input;
  cin >> input;
  
  /**
   * @brief This loop iterates through the input string to process each character.
   */
  for (int i = 0; i < input.size(); i++) {
    // Skip vowels (case-insensitive).
    if (input[i] == 'A' || input[i] == 'O' || input[i] == 'Y' ||
        input[i] == 'E' || input[i] == 'U' || input[i] == 'I')
      continue;
    if (input[i] == 'a' || input[i] == 'o' || input[i] == 'y' ||
        input[i] == 'e' || input[i] == 'u' || input[i] == 'i')
      continue;
      
    // If it's a consonant, process it.
    // Check if the character is already lowercase.
    // 'input[i] - 'A' >= 26' is a way to check if the character is not an
    // uppercase letter. It's not a robust check but works for this problem's constraints.
    if (input[i] - 'A' >= 26)
      cout << '.' << input[i];
    else
      // If it's uppercase, convert to lowercase by adding the difference
      // between 'a' and 'A'.
      cout << '.' << (char)(input[i] + ('a' - 'A'));
  }
  cout << endl;
  return 0;
}