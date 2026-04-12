/**
 * @file 236A.CPP
 * @brief Binary classification based on the cardinality of unique characters in a string.
 * 
 * Logic: Determines the classification of a user ('HER' or 'HIM') by calculating 
 * the number of distinct characters in their identifier. An even count indicates 
 * one class, while an odd count indicates another.
 * 
 * Algorithm: Nested-loop duplicate detection.
 * Time Complexity: $O(N^2)$ where N is the length of the string.
 * Space Complexity: $O(N)$ for the input string buffer.
 */

#include <iostream>

using namespace std;

int main() {
  int len, i, j, duplicate_count = 0, has_duplicate;
  string str;

  // Functional Utility: Reads the user identifier string.
  if (!(cin >> str)) return 0;
  len = (int)str.size();

  /**
   * Block Logic: Identifies characters that appear more than once in the sequence.
   * Invariant: For each character at index i, checks all subsequent characters 
   * to determine if it is a duplicate.
   */
  for (i = 0; i < len; i++) {
    has_duplicate = 0;
    for (j = i + 1; j < len; j++) {
      if (str[i] == str[j]) {
        has_duplicate = 1;
        // Optimization: Breaks early if a duplicate is found for the current index.
        break; 
      }
    }
    // Accumulator: Tracks total characters that are non-unique relative to future indices.
    if (has_duplicate == 1)
      duplicate_count++;
  }

  /**
   * Functional Utility: Cardinality calculation.
   * Logic: Unique Count = Total Length - Sum of characters having subsequent duplicates.
   */
  int unique_count = len - duplicate_count;

  /**
   * Logic: Parity-based binary classification.
   */
  if (unique_count % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  
  return 0;
}
