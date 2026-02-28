/**
 * @file 281A.CPP
 * @brief Solves the "Word Capitalization" problem from Codeforces.
 * @details The problem requires capitalizing the first letter of a given word, leaving the rest of the word unchanged.
 *
 * @algorithm
 * The solution reads the input word as a string. It then inspects the first character.
 * If the first character is a lowercase English letter (by checking if its ASCII value
 * is within the range of 'a' to 'z'), it converts it to its uppercase equivalent.
 * This conversion is done by subtracting 32 from its ASCII value, which is the
 * fixed offset between lowercase and uppercase letters in the ASCII table.
 * Finally, the modified string (with the capitalized first letter) is printed.
 *
 * @complexity
 * Time Complexity: O(L) where L is the length of the string, dominated by the I/O operations. The core logic is O(1).
 * Space Complexity: O(L) to store the input string.
 */
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  /**
   * @variable s
   * @brief Stores the input word.
   */
  string s;

  // Read the word from standard input.
  cin >> s;

  /**
   * @block
   * @brief This block capitalizes the first letter of the word if it's lowercase.
   * @invariant The rest of the string (from the second character onwards) remains unchanged.
   */
  // Pre-condition: Check if the first character s[0] is a lowercase letter.
  // The ASCII value for 'a' is 97. This checks if s[0] is 'a' or comes after it in the ASCII table.
  if ((int)s[0] >= 97)
    // Convert the lowercase character to uppercase by subtracting 32 from its ASCII value.
    s[0] = (int)s[0] - 32;

  // Output the potentially modified string.
  cout << s;
  return 0;
}
