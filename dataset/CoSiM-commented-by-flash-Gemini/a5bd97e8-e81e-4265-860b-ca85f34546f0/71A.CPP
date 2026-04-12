/**
 * @file 71A.CPP
 * @brief Logic for generating word-length-based numeronyms for oversized strings.
 * 
 * Logic: Words exceeding 10 characters are compressed into the format: 
 * (first character + intermediate character count + last character).
 * 
 * Algorithm: Linear scan to determine string length followed by conditional abbreviation.
 * Time Complexity: $O(N \times L)$ where N is the number of test cases and L is the max string length.
 * Space Complexity: $O(L)$ for the character buffer.
 */

#include <iostream>
using namespace std;

int main() {
  // Acts as a character buffer for input words.
  char ch[100];
  int i, j, line, count;
  
  // Reads the total number of test cases (lines) to process.
  cin >> line;

  /**
   * Block Logic: Iteratively processes each input string.
   * Invariant: Processes one word per loop iteration, resetting the character count.
   */
  for (i = 1; i <= line; i++) {
    count = 0;
    cin >> ch;

    /**
     * Block Logic: Calculates the length of the current null-terminated string.
     * Functional Utility: Manual implementation of strlen().
     */
    for (j = 0; ch[j] != '\0'; j++) {
      count++;
    }

    /**
     * Functional Utility: Conditional compression logic.
     * Logic: If the word is longer than 10 characters, it is replaced by its numeronym.
     */
    if (count > 10)
      // Generates numeronym: (First Char) + (Length - 2) + (Last Char).
      cout << ch[0] << count - 2 << ch[count - 1] << endl;
    else
      // Preserves original sequence for short words.
      cout << ch << endl;
  }
  
  return 0;
}
