/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem using ASCII manipulation.
 * @details This program reads a single word and capitalizes its first letter.
 * It checks if the first character is a lowercase letter (between 'a' and 'z')
 * and, if so, converts it to uppercase by subtracting 32 from its ASCII value.
 * This works because the ASCII codes for uppercase and lowercase letters are 32 apart.
 *
 * @algorithm ASCII value modification for case conversion.
 * @complexity
 * Time complexity: O(L) to read the string of length L. The capitalization logic is O(1).
 * Space complexity: O(L) to store the input word.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;
  /**
   * @brief Checks if the first character is a lowercase letter.
   * @invariant The condition `s[0] >= 'a' && s[0] <= 'z'` is true only for lowercase English letters.
   */
  if (s[0] >= 'a' && s[0] <= 'z') {
    // Convert the first character to uppercase by subtracting 32 from its ASCII value.
    s[0] = s[0] - 32;
  }
  cout << s << endl;
  return 0;
}