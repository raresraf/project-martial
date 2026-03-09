/**
 * @file 58A.CPP
 * @brief Subsequence Checker for "hello"
 *        This program determines if the string "hello" can be formed as a
 *        subsequence of a given input string. A subsequence is a sequence
 *        that can be derived from another sequence by deleting some or no
 *        elements without changing the order of the remaining elements.
 *
 * Algorithm: Greedy string matching. It iterates through the input string
 *            and tries to match the characters of "hello" in their sequential order.
 *            It maintains a pointer/index for both the target subsequence ("hello")
 *            and the input string.
 * Time Complexity: O(L_word), where L_word is the length of the input string.
 *                  In the worst case, it iterates through the entire input string once.
 * Space Complexity: O(L_word + L_etalon) for storing both strings.
 */

#include <iostream> // Required for input/output operations (cin, cout)
#include <string>   // Required for string manipulation

using namespace std;

int main() {
  /**
   * Functional Utility: Reads an input word, checks if "hello" is its subsequence,
   *                     and prints "YES" or "NO" accordingly.
   */
  string etalon = "hello", word; // `etalon` is the target subsequence, `word` is the input string.
  cin >> word;                   // Reads the input word from standard input.
  size_t j = 0;                  // `j` is the index for iterating through the `etalon` string.

  /**
   * Block Logic: Iterates through the input `word` to find characters of `etalon`
   *              in their correct sequential order.
   * Pre-condition: `word` is the input string; `etalon` is "hello"; `j` is 0.
   * Invariant: `j` indicates how many characters of `etalon` have been found
   *            as a subsequence in `word` up to the current `i`. The loop
   *            continues as long as there are characters left in `word` AND
   *            characters remaining to be matched in `etalon`.
   */
  for (size_t i = 0; i < word.size() && j < etalon.size(); ++i) {
    // Condition Check: If the current character of `word` matches the
    //                  next required character of `etalon`.
    if (word[i] == etalon[j])
      ++j; // Advance the `etalon` index, indicating a successful match.
  }

  // Functional Utility: Determines the final answer based on whether all characters of `etalon` were matched.
  // If `j` equals `etalon.size()`, it means all characters of "hello" were found in order.
  string answer = ((etalon.size() == j) ? "YES" : "NO");
  cout << answer << endl; // Prints the result ("YES" or "NO") to standard output.
  return 0;               // Indicates successful program termination.
}
