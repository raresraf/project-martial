/**
 * @file 236A.CPP
 * @brief Binary classification based on the cardinality of unique characters in a string.
 * 
 * Logic: Determines the classification of a user ('HER' or 'HIM') by calculating 
 * the number of distinct characters in their identifier (including whitespace). 
 * An odd count results in 'CHAT WITH HER!', an even count in 'IGNORE HIM!'.
 * 
 * Algorithm: Set-based unique character counting via input buffering.
 * Time Complexity: $O(N \log K)$ where N is input length and K is unique character count.
 * Space Complexity: $O(N)$ for the character buffer and unique set.
 */

#include <cstring>
#include <iostream>
#include <set>

using namespace std;

int main() {
  // Data Structure: Set to automatically handle duplicate removal.
  set<char> unique_chars;
  // Buffer for input string (up to 100 characters + terminator).
  char buffer[101];
  int i;

  /**
   * Functional Utility: Reads the entire input line into the buffer.
   * Note: fgets() typically includes the trailing newline character '\n'.
   */
  if (!fgets(buffer, sizeof(buffer), stdin)) return 0;

  /**
   * Block Logic: Populates the set with all characters from the input buffer.
   * Invariant: Each distinct ASCII symbol (including '\n') is represented exactly once.
   */
  for (i = 0; i < (int)strlen(buffer); i++) {
    unique_chars.insert(buffer[i]);
  }

  /**
   * Functional Utility: Parity-based binary classification.
   * Logic: Evaluates the cardinality of the unique set to determine the output class.
   */
  int set_size = unique_chars.size();
  if (set_size % 2 == 0)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
  
  return 0;
}
