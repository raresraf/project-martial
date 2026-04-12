/**
 * @file 71A.CPP
 * @brief Logic for generating word-length-based numeronyms for oversized strings.
 * 
 * Logic: Words exceeding 10 characters are compressed into the format: 
 * (first character + intermediate character count + last character).
 * 
 * Time Complexity: $O(N \times L)$ where N is the word count and L is the string length.
 * Space Complexity: $O(L)$ to store the current processing string.
 */

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
  // Functional Utility: Reads the total number of test cases/words to iterate through.
  int n;
  cin >> n;

  string s;
  /**
   * Block Logic: Main processing loop for input words.
   * Invariant: Each word is evaluated for length-based compression independently.
   */
  for (int i = 0; i < n; ++i) {
    cin >> s;
    
    // Functional Utility: Threshold-based decision point for compression.
    if (s.size() > 10) {
      // Logic: Extracts boundary characters and calculates the middle count via string::size().
      cout << s[0] << s.size() - 2 << s.back();
    } else {
      // Logic: Preserves the original string state for short words.
      cout << s;
    }

    // Functional Utility: Terminates the output for the current word with a newline character.
    cout << '\n';
  }
}
