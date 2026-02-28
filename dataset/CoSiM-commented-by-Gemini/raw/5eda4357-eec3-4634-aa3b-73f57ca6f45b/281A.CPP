/**
 * @file 281A.CPP
 * @brief Solution for the "Word Capitalization" problem from Codeforces.
 * @details This program reads a single word and capitalizes its first letter.
 * It's a very direct and efficient implementation.
 *
 * @algorithm In-place Character Conversion
 * 1. Read the input word into a `std::string`.
 * 2. Use the `toupper()` function to convert the first character of the string
 *    (at index 0) to its uppercase equivalent. The result is assigned back
 *    to the same position, modifying the string in-place.
 * 3. Print the modified string.
 *
 * @complexity
 * - Time: O(L), where L is the length of the word, dominated by reading the string.
 *   The capitalization itself is an O(1) operation.
 * - Space: O(L), to store the input string.
 */

#include <algorithm>
#include <iostream>
#include <string>

// These includes are not used in the final solution.
#include <cmath>
#include <vector>

using namespace std;

/**
 * @brief Unused function to convert a character digit to an integer.
 * This function is not relevant to the problem's solution.
 */
int conver(char a) { return a - 48; }

int main() {
  // `a` stores the input word.
  string a;
  cin >> a;

  /**
   * @brief Capitalize the first letter of the string in-place.
   * `toupper(a[0])` returns the uppercase version of the character at index 0.
   * This works correctly even if the character is already uppercase.
   */
  a[0] = toupper(a[0]);

  // Print the modified string.
  cout << a;

  return 0; // Indicate successful execution.
}
