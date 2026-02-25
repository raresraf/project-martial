/**
 * @file 281A.CPP
 * @brief Solution for a string capitalization problem ("Word Capitalization").
 * @details The program reads a single word and capitalizes its first letter,
 * leaving the rest of the word unchanged.
 *
 * Algorithm:
 * 1. Read the input word as a string.
 * 2. Check if the string is not empty.
 * 3. Access the first character of the string (at index 0).
 * 4. Convert this character to its uppercase equivalent.
 * 5. Print the modified string.
 *
 * The implementation iterates through the string, printing the uppercase version
 * of the first character and the original version of all subsequent characters.
 *
 * Time Complexity: O(L), where L is the length of the string.
 * Space Complexity: O(L) to store the string.
 */
#include <iostream>

using namespace std;

int main() {
  /**
   * @note These lines are a common optimization in competitive programming to
   * speed up C++ streams by decoupling them from the C standard streams.
   */
  ios::sync_with_stdio(0);
  cin.tie(0);

  string s;
  cin >> s;

  /**
   * @block
   * @description This loop iterates through the input string character by character.
   * It capitalizes the first character and prints the rest as is.
   */
  for (int i = 0; i < s.length(); i++) {
    // Check if the current character is the first one in the string.
    if (i == 0) {
      // Capitalize and print the first character.
      cout << char(toupper(s[i]));
    } else {
      // Print the remaining characters without modification.
      cout << s[i];
    }
  }

  cout << endl;

  return 0;
}