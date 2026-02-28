/**
 * @file 281A.CPP
 * @brief Solution for the "Word Capitalization" problem from Codeforces.
 * @details This program reads a single word and capitalizes its first letter. If the
 * first letter is already capitalized, the word remains unchanged.
 *
 * @algorithm String Manipulation
 * 1. Read the input word as a string.
 * 2. Get the uppercase equivalent of the first character.
 * 3. Modify the first character of the string to be its uppercase version.
 * 4. Print the resulting string.
 *
 * @complexity
 * - Time: O(L), where L is the length of the word, dominated by the time to read
 *   the string from standard input. The capitalization itself is O(1).
 * - Space: O(L), to store the word in a string.
 */

#include <cstring>   // For toupper
#include <iostream>
#include <string>    // For std::string

using namespace std;

int main() {
  // s: stores the input word.
  string s;
  cin >> s;

  // `upper` holds the integer representation of the uppercase version of the first character.
  int upper = toupper(s[0]);

  /**
   * @brief This block checks if the first letter is already uppercase.
   * Note: This check is redundant. A simpler, unconditional assignment
   * `s[0] = toupper(s[0]);` would produce the same correct result with less code.
   */
  // Pre-condition: Check if the character is already uppercase.
  if (s[0] == upper) {
    // If it is, print the string as is.
    cout << s << '\n';
  } else {
    // If it's not, update the first character to its uppercase equivalent.
    s[0] = upper;
    cout << s << '\n';
  }
  
  return 0; // Indicate successful execution.
}
