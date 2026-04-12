/**
 * @file 71A.CPP
 * @brief Logic for generating word-length-based numeronyms for oversized strings.
 * 
 * Logic: Words exceeding 10 characters are compressed into the format: 
 * (first character + intermediate character count + last character). 
 * Shorter words are preserved in their original form.
 * 
 * Algorithm: Single-pass scan with length-based conditional abbreviation.
 * Time Complexity: $O(N \times L)$ where N is the number of test cases and L is the max string length.
 * Space Complexity: $O(L)$ for the fixed-size character buffer.
 */

#include <iostream>
#include <string.h>

using namespace std;

int main() {
  int n, len;
  // Static buffer for input words.
  char str[102];
  
  // Functional Utility: Reads the total number of independent words to process.
  cin >> n;

  /**
   * Block Logic: Main processing loop for input words.
   * Invariant: Each word is processed independently, resetting the local length variable.
   */
  while (n--) {
    cin >> str;
    len = strlen(str);
    
    /**
     * Functional Utility: Conditional compression logic.
     */
    if (len <= 10)
      // Logic: Preserve state for short words.
      cout << str << endl;
    else {
      /**
       * Logic: Abbreviation construction.
       * Extracts boundary characters and calculates middle count via (length - 2).
       */
      cout << str[0] << len - 2 << str[len - 1] << endl;
    }
  }
  
  return 0;
}
