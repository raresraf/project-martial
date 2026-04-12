/**
 * @file 71A.CPP
 * @brief Logic for generating length-based numeronyms for oversized strings.
 * 
 * Logic: Compresses words exceeding 10 characters into a format: 
 * (first character + count of intermediate characters + last character).
 * 
 * Algorithm: Buffering of input strings followed by conditional abbreviation.
 * Time Complexity: $O(N \times L)$ where N is the word count and L is the average length.
 * Space Complexity: $O(N \times L)$ due to vector-based buffering of all input words.
 */

#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  int n;
  // Data Structure: Buffers all input strings before processing.
  vector<string> words;
  string str;

  // Functional Utility: Reads total word count and populates the buffer.
  cin >> n;
  while (n--) {
    cin >> str;
    words.push_back(str);
  }

  /**
   * Block Logic: Main processing loop for buffered words.
   * Invariant: Each word is independently evaluated for length-based compression.
   */
  for (const string s : words) {
    /**
     * Functional Utility: Conditional compression logic.
     * Logic: Threshold for abbreviation is strictly > 10 characters.
     */
    if (s.length() > 10) {
      // Generates numeronym: (First char) + (Length - 2) + (Last char).
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      // Logic: Preserves original sequence for short words.
      cout << s << endl;
    }
  }

  return 0;
}
