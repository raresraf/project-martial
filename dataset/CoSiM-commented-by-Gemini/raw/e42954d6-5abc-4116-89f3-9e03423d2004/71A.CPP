/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 * @description This program abbreviates words that are longer than 10 characters.
 *              The abbreviation consists of the first character, the number of
 *              characters between the first and last, and the last character.
 *
 * Algorithm: The program reads the number of test cases, `t`. It then enters a
 *            loop that runs `t` times. In each iteration, it reads a word,
 *            checks if its length is greater than 10, and prints either the
 *            original word or its abbreviated form accordingly.
 *
 * Time Complexity: O(T * L), where T is the number of words and L is the maximum
 *                  length of a word.
 * Space Complexity: O(L) to store the longest word read.
 */
#include <iostream>
#include <string.h> // C-style string header, <string> is more conventional for C++.

using namespace std;

// Using main() without an explicit int return type is non-standard but often compiled.
main() {
  string word;
  int t; // Number of test cases (words).
  cin >> t;

  /**
   * Block Logic: Loop `t` times to process each word.
   */
  while (t > 0) {
    cin >> word;
    int n = word.length();
    
    // Pre-condition: Check if the word's length exceeds the abbreviation threshold.
    if (word.length() <= 10) {
      // If the word is not "too long", print it as is.
      cout << word << endl;
    } else {
      // Otherwise, print the abbreviated version.
      cout << word[0] << n - 2 << word[n - 1] << endl;
    }
    t = t - 1; // Decrement the loop counter.
  }
}
