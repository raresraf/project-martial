/**
 * @file 281A.CPP
 * @brief Solution for the "Word Capitalization" problem.
 *
 * @details
 * This program reads a single word and capitalizes its first letter. If the
 * first letter is already capitalized or is not an alphabet character, it
 * remains unchanged. The rest of the word is not affected.
 *
 * @section
 * Algorithm: Direct in-place modification of the first character.
 * Time Complexity: O(1) for the capitalization logic itself, plus O(N) for string input/output, where N is the length of the string.
 * Space Complexity: O(N) to store the input string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // 's' will store the input word.
  string s;

  // Read a single word from standard input.
  cin >> s;

  /**
   * @brief Capitalize the first character of the string.
   *
   * Logic: The `toupper()` function from the standard library is used to convert
   * the character at index 0 to its uppercase equivalent. The modification
   * happens in-place.
   */
  s[0] = toupper(s[0]);

  // Print the modified string to standard output.
  cout << s;

  return 0;
}
