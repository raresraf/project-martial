/**
 * @file 236A.CPP
 * @brief Binary classification based on the cardinality of unique characters in a string.
 * 
 * Logic: Determines the classification of a user ('HER' or 'HIM') by calculating 
 * the number of distinct characters in their identifier. An even count indicates 
 * one class, while an odd count indicates another.
 * 
 * Algorithm: Set-based unique character counting.
 * Time Complexity: $O(N \log K)$ where N is string length and K is unique character count.
 * Space Complexity: $O(K)$ to store the set of unique characters.
 */

#include <iostream>
#include <set>
#include <string>

using namespace std;

int main() {
  string str;
  // Data Structure: Set to automatically handle duplicate removal and track distinct symbols.
  set<char> distinct_chars;

  // Functional Utility: Reads the input character sequence.
  if (!(cin >> str)) return 0;

  /**
   * Block Logic: Populates the set with characters from the input string.
   * Invariant: After iteration, 'distinct_chars' contains exactly the set of 
   * unique symbols present in 'str'.
   */
  for (int i = 0; i < (int)str.length(); ++i) {
    distinct_chars.insert(str[i]);
  }

  /**
   * Functional Utility: Final state evaluation.
   * Logic: Maps the parity of the unique set size to the output classification.
   */
  if (distinct_chars.size() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
    return 0;
  }

  // Fallback: Odd cardinality condition.
  cout << "IGNORE HIM!" << endl;

  return 0;
}
