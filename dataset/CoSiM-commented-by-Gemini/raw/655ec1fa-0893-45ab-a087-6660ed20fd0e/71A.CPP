/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem on Codeforces (Problem 71A).
 * @details This program reads a number of words and for each word, it checks its length.
 * If the length is greater than 10, it prints an abbreviation consisting of the
 * first character, the number of characters in between, and the last character.
 * Otherwise, it prints the word as is. This is a common introductory problem
 * focusing on basic string manipulation and I/O.
 *
 * Algorithm: String Abbreviation based on length.
 * Time Complexity: O(N), where N is the number of words to process. Each word is processed in constant time relative to its length after being read.
 * Space Complexity: O(S), where S is the total size of all input strings stored in the vector.
 */
// Codeforces
// 71A - Way too long word

#include <iostream>
#include <vector>
using namespace std;

int main() {
  int size;
  // Pre-condition: Reads the total number of words that will be processed.
  cin >> size;

  vector<string> vs;
  string line;

  /**
   * @brief Block Logic: Reads all words from standard input and stores them.
   * Invariant: After this loop, the `vs` vector contains `size` words.
   */
  for (int i = 0; i < size; i++) {
    cin >> line;
    vs.push_back(line);
  }

  /**
   * @brief Block Logic: Iterates through the stored words and applies the abbreviation rule.
   * Invariant: Processes one word per iteration, printing either the original or abbreviated form.
   */
  for (int i = 0; i < size; i++) {
    // Pre-condition: Checks if the word's length exceeds the threshold of 10 characters.
    if (vs[i].size() <= 10)
      // If the word is not "too long", print it without modification.
      cout << vs[i] << endl;
    else {
      // If the word is "too long", construct and print the abbreviation.
      // The format is: first character, count of middle characters, last character.
      cout << vs[i][0];
      cout << (vs[i].size() - 2);
      cout << vs[i][vs[i].size() - 1] << endl;
    }
  }

  return 0;
}
