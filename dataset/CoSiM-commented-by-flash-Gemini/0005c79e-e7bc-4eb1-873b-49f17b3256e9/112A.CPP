/**
 * @0005c79e-e7bc-4eb1-873b-49f17b3256e9/112A.CPP
 * @brief Implements a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them to lowercase, and then
 * compares them character by character to determine their lexicographical order.
 *
 * Algorithm: Iterative character-by-character comparison after converting to lowercase.
 * Time Complexity: O(N) where N is the length of the shorter string, as it involves
 *                  iterating through characters of both strings up to the length of the
 *                  shorter one. The `convert_lower` function also contributes O(N) for
 *                  each string.
 * Space Complexity: O(N) for storing the input strings and their lowercase versions.
 */

#include <cstdio>
#include <iostream>
#include <string.h>

using namespace std;

/**
 * @brief Converts a given string to its lowercase equivalent.
 * @param ch The input string to convert.
 * @return The lowercase version of the input string.
 *
 * Functional Utility: Ensures case-insensitive comparison by normalizing string case.
 * Invariant: The returned string will contain only lowercase characters.
 */
string convert_lower(string ch) {
  // Iterates through each character of the string.
  for (int i = 0; i < ch.size(); i++) {
    // Converts the current character to lowercase.
    ch[i] = tolower(ch[i]);
  }

  return ch;
}

int main() {
  string first, second;
  /**
   * Functional Utility: Acquires two strings from standard input for subsequent
   * case-insensitive lexicographical comparison.
   */
  cin >> first >> second;
  
  /**
   * Functional Utility: Normalizes the input strings to lowercase to facilitate
   * a case-insensitive lexicographical comparison as per problem requirements.
   * Invariant: Both 'first' and 'second' strings will contain only lowercase characters
   *            after this operation.
   */
  first = convert_lower(first);
  second = convert_lower(second);
  
  /**
   * Functional Utility: Initializes 'res' to store the outcome of the lexicographical comparison.
   * Values: 0 for equality, 1 if 'first' is lexicographically greater, -1 if 'first' is lexicographically smaller.
   */
  int res = 0;
  
    /**
     * Block Logic: Iterates through the characters of both strings to perform
     * a lexicographical comparison. The loop continues as long as characters match
     * or until a definitive difference is found, or the end of the shorter string is reached.
     * Pre-condition: Both 'first' and 'second' strings are already converted to lowercase.
     * Invariant: At the start of each iteration, 'res' holds the lexicographical
     *            comparison result for the substring examined so far.
     */
    for (int i = 0; i < first.size(); i++) {
      /**
       * Pre-condition: Current characters at index 'i' from both strings are being compared.
       * Block Logic: Checks if characters at the current index are identical.
       */
      if (first[i] == second[i]) {
        // Functional Utility: Continues comparison with the next character if the current characters are identical.
        continue;
      }
      /**
       * Pre-condition: Characters at index 'i' are different.
       * Block Logic: Determines if the character in 'first' is lexicographically greater
       *              than the character in 'second'.
       */
      else if (first[i] > second[i]) {
        // Functional Utility: Sets 'res' to indicate 'first' is greater and terminates comparison.
        res = 1;
        break; // Optimization: Early exit once lexicographical order is established.
      }
      /**
       * Pre-condition: Characters at index 'i' are different, and 'first[i]' is not greater than 'second[i]'.
       * Block Logic: Infers that the character in 'first' must be lexicographically smaller
       *              than the character in 'second'.
       */
      else {
        // Functional Utility: Sets 'res' to indicate 'first' is smaller and terminates comparison.
        res = -1;
        break; // Optimization: Early exit once lexicographical order is established.
      }
    }  
  /**
   * Functional Utility: Outputs the integer result of the lexicographical comparison
   * to standard output, indicating the relative order of the two input strings.
   */
  cout << res;
  return 0;
}
