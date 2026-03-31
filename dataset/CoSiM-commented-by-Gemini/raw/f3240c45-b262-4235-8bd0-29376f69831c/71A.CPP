/**
 * @file 71A.CPP
 * @brief Abbreviates words longer than 10 characters using a two-pass approach.
 *
 * This program solves the "Way Too Long Words" problem. It first reads all
 * input words into a vector. Then, in a second pass, it iterates through the
 * vector. For each word, if its length is greater than 10, it prints an
 * abbreviation (first character, number of middle characters, last character).
 * Otherwise, it prints the original word.
 *
 * @author f3240c45-b262-4235-8bd0-29376f69831c
 *
 * @algorithm Two-Pass String Abbreviation
 * @complexity
 * Time: O(N * L), where N is the number of words and L is their average length,
 *       dominated by reading and processing the strings.
 * Space: O(N * L), to store all N words in a vector.
 */
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  int n;
  cin >> n;
  // A vector to store all the words before processing.
  vector<string> words;

  /**
   * @brief First pass: Read all words into the vector.
   */
  for (int i = 0; i < n; i++) {
    string word;
    cin >> word;
    words.push_back(word);
  }

  /**
   * @brief Second pass: Iterate through the stored words and print the result.
   * This uses a range-based for loop, a modern C++ feature.
   */
  for (string &word : words) {
    int s = word.size();
    /**
     * @brief Check if the word needs to be abbreviated.
     */
    if (s <= 10) {
      cout << word << endl;
    } else {
      // `val` holds the number of characters between the first and last.
      int val = s - 2;
      // `cout` handles the conversion of the integer `val` to a string.
      cout << word[0] << val << word[s - 1] << endl;
    }
  }

  return 0;
}