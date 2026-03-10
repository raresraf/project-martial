/**
 * @file 281A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Word Capitalization").
 * This program capitalizes the first letter of a given word.
 *
 * @algorithm Direct character manipulation by modifying ASCII values.
 * @time_complexity O(L), where L is the length of the string, dominated by the I/O.
 * The capitalization logic is O(1).
 * @space_complexity O(L) to store the input string.
 */
#include <iostream>

using namespace std;

int main() {
  string input;
  cin >> input;
  /**
   * @block
   * @brief Capitalizes the first letter if it is lowercase.
   *
   * Pre-condition: Checks if the string is non-empty and if the ASCII value
   * of the first character is greater than 96 (the value for '`'), which
   * effectively checks if it is a lowercase letter.
   */
  if (input.length() > 0 && input[0] > 96) {
    // Converts the character to uppercase by subtracting 32 from its ASCII value.
    input[0] = input[0] - 32;
  }
  cout << input << endl;
  return 0;
}