/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem by processing words one by one.
 * @author Sagindyk Urazayev
 *
 * This program abbreviates words longer than 10 characters. This implementation
 * is memory-efficient as it processes each word as it's read from the input
 * stream, without storing the entire list of words in memory.
 *
 * @b Algorithm: For each word, if its length exceeds 10, it's abbreviated to
 *    its first letter, the count of its middle letters, and its last letter.
 *    Otherwise, the word is printed unchanged.
 * @b Time_Complexity: O(N * L), where N is the number of words and L is their
 *    maximum length, dominated by I/O operations.
 * @b Space_Complexity: O(L), as only one word needs to be stored in memory at a time.
 */
// Sagindyk Urazayev
// codeforces.com/problemset/problem/71/A
#include <iostream>
#include <string>

using namespace std;

int main() {
  int i, n;
  cin >> n;
  
  // Basic input validation for the number of words.
  if ((n < 1) || (n > 100)) {
    cout << "Wrong value!" << endl;
  }
  
  string word;
  
  /**
   * @brief Main loop to read and process each word sequentially.
   * This stream-based approach is memory efficient.
   */
  for (i = 0; i < n; i++) {
    cin >> word;
    // Check if the word's length is greater than 10.
    if (word.size() > 10) {
      // Construct and print the abbreviation directly.
      // `cout` automatically converts the integer (word.size() - 2) to a string.
      cout << word.at(0) << word.size() - 2 << word.at(word.size() - 1) << endl;
    } else {
      // If the word is not too long, print it as is.
      cout << word << endl;
    }
  }
  return 0;
}