/**
 * @file 58A.CPP
 * @brief Subsequence detection algorithm for identifying target patterns in noisy character streams.
 * 
 * Logic: Determines if the word "hello" exists as a subsequence within the input 
 * string 'str'. This requires the characters 'h', 'e', 'l', 'l', 'o' to appear 
 * in that specific relative order.
 * 
 * Algorithm: Greedy linear scan with a monotonic pattern pointer.
 * Time Complexity: $O(N)$ where N is the length of the input string.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  // str: Input stream, str1: Target pattern.
  string str, str1 = "hello";
  if (!(cin >> str)) return 0;

  // j: current index in target pattern, pass: count of matched characters.
  int j = 0, pass = 0;

  /**
   * Block Logic: Greedily consumes input characters to satisfy the target pattern.
   * Invariant: 'pass' tracks the length of the longest matched prefix of "hello".
   */
  for (int i = 0; i < (int)str.length(); i++) {
    /**
     * Logic: If current input character matches the next required pattern character.
     */
    if (str[i] == str1[j]) {
      j++;    // Advance pattern expectation.
      pass++; // Increment success counter.

      /**
       * Functional Utility: Early termination.
       * Logic: Once all 5 characters of "hello" are matched in order, the search is complete.
       */
      if (pass == 5)
        break;
    }
  }

  /**
   * Functional Utility: Final state classification.
   */
  if (pass == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
