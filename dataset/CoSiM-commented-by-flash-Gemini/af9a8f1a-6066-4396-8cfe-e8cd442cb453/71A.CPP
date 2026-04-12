/**
 * @file 71A.CPP
 * @brief Logic for generating word-length-based numeronyms for oversized strings.
 * 
 * Logic: Words exceeding 10 characters are compressed into the format: 
 * (first character + count of internal characters + last character). 
 * Shorter words are preserved in their original state.
 * 
 * Algorithm: Buffering of input strings followed by iterative transformation.
 * Time Complexity: $O(N \times L)$ where N is the word count and L is the average string length.
 * Space Complexity: $O(N \times L)$ due to the buffering of all input strings.
 */

#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief Functional Utility: Processes and outputs a single word according to length constraints.
 * @param s The source string to be evaluated.
 * Logic: Uses a vector-based character counting approach to determine string length.
 */
void show(string s) {
  // Data Structure: Temporary buffer to calculate the cardinality of characters.
  vector<char> vec;
  for (auto c : s)
    vec.push_back(c);

  /**
   * Functional Utility: Conditional compression logic.
   * Logic: Threshold for abbreviation is strictly > 10 characters.
   */
  if (vec.size() > 10)
    // Generates numeronym: (First char) + (Internal count) + (Last char).
    cout << s[0] << vec.size() - 2 << s[vec.size() - 1] << endl;
  else
    // Logic: Preserves the original sequence state.
    cout << s << endl;
}

int main() {
  // Data Structure: Buffer for the entire set of input words.
  string arr[500];
  int num;
  
  // Functional Utility: Reads total word count and populates the buffer.
  cin >> num;
  for (int i = 0; i < num; i++) {
    cin >> arr[i];
  }

  /**
   * Block Logic: Iteratively invokes the transformation logic for each buffered word.
   */
  for (int i = 0; i < num; i++) {
    show(arr[i]);
  }
  
  return 0;
}
