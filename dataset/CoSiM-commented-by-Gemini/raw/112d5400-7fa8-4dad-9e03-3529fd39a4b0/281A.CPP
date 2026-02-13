/**
 * @file 281A.CPP
 * @brief Solution for the "Word Capitalization" problem (Codeforces 281A).
 * @author competitive-programmer
 *
 * @details This program takes a single word as input and capitalizes its first
 * letter. If the first letter is already capitalized, it remains unchanged.
 *
 * Algorithm:
 * 1. Read the input word into a string.
 * 2. Access the first character of the string (at index 0).
 * 3. Use the `toupper()` function to convert this character to its uppercase
 *    equivalent.
 * 4. Update the first character of the string with the new uppercase character.
 * 5. Print the modified string.
 *
 * Time Complexity: O(1), as the core logic only operates on the first character.
 * The input reading is O(L) where L is the length of the word.
 * Space Complexity: O(L) to store the word.
 */
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  string line;
  // Read the word from standard input.
  cin >> line;

  // Access the first character (at index 0) and convert it to uppercase.
  // The toupper() function handles cases where the character is already
  // uppercase or not a letter.
  line[0] = toupper(line[0]);
  
  // Print the modified string.
  cout << line;
  
  return 0;
}
