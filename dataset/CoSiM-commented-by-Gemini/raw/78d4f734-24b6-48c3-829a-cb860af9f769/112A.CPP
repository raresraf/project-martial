/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem with a minor bug.
 *
 * This program compares two strings lexicographically, ignoring differences in case.
 * It should output -1 if the first string is lexicographically smaller than the second,
 * 1 if it is larger, and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings, `word1` and `word2`.
 * 2. Convert `word1` to all lowercase letters.
 * 3. Convert `word2` to all lowercase letters.
 * 4. Perform a lexicographical comparison on the two lowercase strings.
 * 5. Print the result according to the problem's specification (-1, 1, or 0).
 *
 * @implementation_details
 * - The code uses ASCII value manipulation (`*it = *it + 32`) to convert characters
 *   to lowercase. This works for English letters but is not a robust, locale-safe method.
 *   Using `std::transform` with `::tolower` is preferred.
 * - The standard `string::compare` function is used, which returns a negative, zero,
 *   or positive value.
 *
 * @bug
 * The final output logic is flawed. The problem requires an output of exactly -1, 0, or 1.
 * While this code correctly outputs -1 and 0, if the first string is greater than the
 * second, `word1.compare(word2)` returns an arbitrary positive integer, not necessarily 1.
 * The code prints this arbitrary positive value directly, which does not conform to the
 * problem specification. A correct implementation would check `if (result > 0)` and
 * print `1`.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings. The loops for
 * converting to lowercase and the final comparison both take linear time.
 * Space Complexity: O(L) to store the two strings.
 */
#include <iostream>
#include <string> // It's good practice to include the string header
#include <algorithm> // For a more robust transform/tolower solution

using namespace std;

int main() {
  string word1, word2;
  int result;

  getline(cin, word1);
  getline(cin, word2);

  /**
   * @brief Convert both strings to lowercase.
   * This method assumes ASCII and that any char code < 97 is an uppercase letter.
   */
  for (auto it = word1.begin(); it != word1.end(); it++) {
    if (*it < 'a') // Using 'a' is more readable than the magic number 97
      *it = *it + 32;
  }
  for (auto it = word2.begin(); it != word2.end(); it++) {
    if (*it < 'a')
      *it = *it + 32;
  }

  // Perform lexicographical comparison on the lowercase strings.
  result = word1.compare(word2);

  /**
   * @brief Print the result, with a bug in the "greater than" case.
   * If result is positive, it should print 1, but it prints the raw positive value.
   */
  if (result > 0)
    cout << 1 << endl;
  else if (result < 0)
    cout << -1 << endl;
  else
    cout << 0 << endl;

  return 0;
}
