/**
 * @file 112A.CPP
 * @brief Solution to Codeforces problem 112A - Petya and Strings.
 *        This program performs a case-insensitive lexicographical comparison of two input strings.
 * Algorithm:
 *   1. Read two input strings, `word1` and `word2`.
 *   2. Convert all characters in both strings to lowercase. This ensures a case-insensitive comparison.
 *      Characters with ASCII values less than 97 (i.e., uppercase letters) are converted by adding 32.
 *   3. Compare the two processed strings lexicographically using `string::compare`.
 *   4. The problem typically expects output: -1 if `word1` < `word2`, 0 if `word1` == `word2`, and 1 if `word1` > `word2`.
 *      The current implementation prints the direct result of `string::compare` if it's non-negative,
 *      and -1 if `string::compare` returns a negative value. This aligns with printing -1 for `word1 < word2`
 *      and 0 for `word1 == word2`, but prints the raw positive difference for `word1 > word2`.
 * Time Complexity: O(L), where L is the maximum length of the two input strings, due to character iteration and comparison.
 * Space Complexity: O(L) for storing the input strings.
 */
#include <iostream>

using namespace std;

int main() {
  /**
   * @brief Main function to solve the "Petya and Strings" problem.
   *
   * Reads two strings, converts them to lowercase, and then compares them
   * lexicographically. Prints the comparison result based on specific rules.
   * Note: The output format for greater-than comparisons might differ from typical
   * competitive programming expectations, as it prints the raw positive difference
   * from `string::compare` rather than a standardized '1'.
   * @return int Returns 0 upon successful execution.
   */
  string word1, word2;
  int result;

  getline(cin, word1); // Read the first word from standard input.
  getline(cin, word2); // Read the second word from standard input.

  // Block Logic: Converts all characters in word1 to lowercase for case-insensitive comparison.
  // Pre-condition: 'word1' contains characters that might be uppercase.
  // Invariant: After the loop, all uppercase letters in 'word1' are converted to lowercase.
  for (auto it = word1.begin(); it != word1.end(); it++) {
    // Inline: Checks if the character is an uppercase letter (ASCII value less than 97).
    if (*it < 97)
      // Inline: Converts uppercase character to lowercase by adding 32 to its ASCII value.
      *it = *it + 32;
  }
  // Block Logic: Converts all characters in word2 to lowercase for case-insensitive comparison.
  // Pre-condition: 'word2' contains characters that might be uppercase.
  // Invariant: After the loop, all uppercase letters in 'word2' are converted to lowercase.
  for (auto it = word2.begin(); it != word2.end(); it++) {
    // Inline: Checks if the character is an uppercase letter.
    if (*it < 97)
      // Inline: Converts uppercase character to lowercase.
      *it = *it + 32;
  }

  // Block Logic: Performs a lexicographical comparison between the two normalized strings.
  // Outcome: 'result' is 0 if strings are equal, positive if word1 > word2, negative if word1 < word2.
  result = word1.compare(word2);
  // Block Logic: Prints the comparison result according to problem-specific rules.
  // Pre-condition: 'result' holds the outcome of the lexicographical comparison.
  // Post-condition: Prints the result directly if non-negative, otherwise prints -1.
  if (result >= 0)
    cout << result << endl; // Prints 0 if equal, or the positive difference if word1 > word2.
  else
    cout << -1 << endl; // Prints -1 if word1 < word2.

  return 0;
}
