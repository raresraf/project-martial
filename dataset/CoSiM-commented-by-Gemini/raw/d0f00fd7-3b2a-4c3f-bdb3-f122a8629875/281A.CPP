/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given word.
 *
 * This program solves the "Word Capitalization" problem. It reads a single
 * word and ensures its first character is uppercase, then prints the result.
 *
 * @b Algorithm: Direct Character Manipulation
 *   The first character of the string is accessed directly. Its case is
 *   converted to uppercase using arithmetic operations on its ASCII value.
 *
 * @b Time_Complexity: O(L) for reading the string of length L. The
 *   capitalization operation itself is O(1).
 * @b Space_Complexity: O(L) to store the input string.
 */
#include "bits/stdc++.h"
using namespace std;

int main(int argc, char const *argv[]) {
  string str;
  cin >> str;

  /**
   * @brief Manually capitalize the first character.
   * This logic is equivalent to `toupper(str[0])`. It works by:
   * 1. Converting the character to lowercase (`tolower`).
   * 2. Finding its 0-indexed position in the alphabet (`- 'a'`).
   * 3. Converting that index back to an uppercase character (`+ 'A'`).
   */
  str[0] = (tolower(str[0]) - 'a' + 'A');
  
  // Print the modified string.
  cout << str;
  return 0;
}
