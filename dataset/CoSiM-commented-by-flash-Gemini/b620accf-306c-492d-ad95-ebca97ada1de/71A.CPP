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
#include <string>
using namespace std;

int main() {
  string s;
  int n;

  // Functional Utility: Reads the total number of independent words to process.
  if (!(cin >> n)) return 0;

  /**
   * Block Logic: Main processing loop for input words.
   * Constraint: Processes up to 100 test cases as per domain-specific bounds.
   */
  if (n >= 1 && n <= 100) {
    for (int i = 0; i < n; i++) {
      cin >> s;
      
      /**
       * Functional Utility: Conditional compression logic.
       * Logic: Threshold for abbreviation is strictly > 10 characters.
       */
      if (s.length() > 10) {
        /**
         * Logic: Abbreviation construction.
         * Extracts boundary characters and calculates middle count via (length - 2).
         */
        cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
      } else {
        // Logic: Preserves original sequence state for short words.
        cout << s << endl;
      }
    }
  }
  
  return 0;
}
