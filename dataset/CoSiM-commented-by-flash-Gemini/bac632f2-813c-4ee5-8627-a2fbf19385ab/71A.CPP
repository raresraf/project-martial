/**
 * @file 71A.CPP
 * @brief Logic for generating word-length-based numeronyms for oversized strings.
 * 
 * Logic: Words exceeding 10 characters are compressed into the format: 
 * (first character + count of internal characters + last character). 
 * Shorter words are preserved in their original form.
 * 
 * Algorithm: Single-pass scan with length-based conditional abbreviation.
 * Time Complexity: $O(N \times L)$ where N is the number of test cases and L is the max string length.
 * Space Complexity: $O(L)$ for the string buffer.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  int t;
  // Functional Utility: Reads the total number of independent words to process.
  if (!(cin >> t)) return 0;

  /**
   * Block Logic: Main processing loop for input words.
   * Invariant: Each word is processed independently, resetting the local length variable.
   */
  while (t--) {
    string str;
    cin >> str;
    int n = (int)str.length();
    
    /**
     * Functional Utility: Conditional compression logic.
     */
    if (n <= 10) {
      // Logic: Preserve state for short words.
      cout << str << '\n';
    } else {
      /**
       * Logic: Abbreviation construction.
       * Extracts boundary characters and calculates middle count via (length - 2).
       */
      cout << str[0] << n - 2 << str[n - 1] << '\n';
    }
  }
  
  return 0;
}
