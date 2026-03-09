/**
 * @file 281A.CPP
 * @brief Capitalizes the first letter of a given word.
 *
 * This program solves the "Word Capitalization" problem from Codeforces.
 * It reads a single word and ensures that its first letter is uppercase.
 * If the first letter is already uppercase or not an alphabet, it remains unchanged.
 *
 * @algorithm String Manipulation
 * @complexity
 * - Time: O(1), as it only inspects and potentially modifies the first character.
 *         The time to read and print the string is O(L), where L is the string length.
 * - Space: O(L) to store the input string. The modification is done in-place.
 */
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  string text;
  cin >> text;

  /**
   * @brief Block Logic: Capitalizes the first character if it is a lowercase letter.
   * Note: This method of capitalization by subtracting 32 is common in competitive
   * programming but relies on the ASCII character set. A more portable approach
   * would be to use the `toupper()` function from the `<cctype>` header.
   */
  // Pre-condition: Check if the first character is a lowercase English letter.
  if (text[0] >= 'a' && text[0] <= 'z') {
    // Inline: Convert to uppercase by subtracting the ASCII difference.
    text[0] -= 32;
  }

  cout << text << endl;

  return 0;
}