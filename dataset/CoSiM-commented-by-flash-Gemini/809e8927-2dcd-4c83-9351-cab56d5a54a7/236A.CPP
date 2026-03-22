/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *        This program determines if a given username belongs to a "boy" or a "girl"
 *        based on the parity (even or odd) of the number of distinct characters in the username.
 *        If the number of distinct characters is even, it's a "girl" ("CHAT WITH HER!").
 *        If the number of distinct characters is odd, it's a "boy" ("IGNORE HIM!").
 *        This implementation uses an in-place string modification approach to count distinct characters.
 *
 * Algorithm: In-place Duplicate Character Removal followed by Parity Check
 *   1. Reads the input username string `name`.
 *   2. Iterates through the string using nested loops:
 *      a. The outer loop selects a character `name.at(i)`.
 *      b. The inner loop searches for duplicates of `name.at(i)` starting from `i+1`.
 *      c. If a duplicate `name.at(j)` is found:
 *         i. It is removed from the string using `name.erase(j, 1)`.
 *         ii. The inner loop index `j` is decremented to re-check the character that
 *             shifted into the current position.
 *         iii. The effective length `length` is decremented.
 *   3. After processing, `name.length()` represents the count of distinct characters.
 *   4. Checks the parity of `name.length()`:
 *      a. If even, prints "CHAT WITH HER!".
 *      b. If odd, prints "IGNORE HIM!".
 *
 * Time Complexity: O(L^2), where L is the original length of the input string `name`.
 *                  This is due to the nested loops and the `string::erase` operation,
 *                  which can take O(L) time in the worst case for each removal.
 * Space Complexity: O(L) for storing the input string `name`. O(1) for auxiliary variables.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Boy or Girl" problem.
   *
   * Reads a username, removes duplicate characters in-place, and then determines
   * if the resulting number of distinct characters is even or odd to print the message.
   *
   * @return int Returns 0 upon successful execution.
   */
  string name; // Input username string.
  cin >> name;
  int length = name.length(); // Current length of the string, updated as characters are removed.
  /**
   * Block Logic: Iterates through the string to find and remove duplicate characters.
   * This modifies the string `name` in-place, leaving only distinct characters.
   * Invariant: After the outer loop completes, `name` contains only unique characters.
   */
  for (int i = 0; i < length; i++) {
    char temp = name.at(i); // Character to compare against for duplicates.
    /**
     * Block Logic: Inner loop to find duplicates of `temp` starting from `i + 1`.
     * If a duplicate is found, it's erased, and loop variables are adjusted.
     */
    for (int j = i + 1; j < length; j++) {
      char holder = name.at(j); // Character to compare with `temp`.
      /**
       * Conditional Logic: Checks if `holder` is a duplicate of `temp`.
       */
      if (holder == temp) {
        name.erase(j, 1); // Inline: Erases 1 character at index `j`.
        j--;              // Inline: Decrement `j` to re-check the character that shifted into this position.
        length--;         // Inline: Decrement `length` as the string has shrunk.
      }
    }
  }
  /**
   * Conditional Logic: Checks the parity of the length of the string with distinct characters.
   * Prints "CHAT WITH HER!" if even, "IGNORE HIM!" if odd.
   */
  if (name.length() % 2 == 0) { // If the count of distinct characters is even.
    cout << "CHAT WITH HER!";
  } else { // If the count of distinct characters is odd.
    cout << "IGNORE HIM!";
  }
  return 0;
}
