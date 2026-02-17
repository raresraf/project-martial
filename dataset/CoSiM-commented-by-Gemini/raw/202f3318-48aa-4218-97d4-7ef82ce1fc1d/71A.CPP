/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" competitive programming problem.
 *
 * @details This program implements a text abbreviation rule. For a given list of
 * words, it checks if a word's length is greater than 10 characters. If it is,
 * the word is abbreviated to its first letter, the count of letters between the
 * first and last, and its last letter. Otherwise, the word is printed as is.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the number of words to process, `n`.
 * 2. Loop `n` times to process each word.
 * 3. Inside the loop, read a word.
 * 4. Check the length of the word.
 * 5. If the length is greater than 10:
 *    a. Print the first character.
 *    b. Print the length of the word minus 2.
 *    c. Print the last character.
 * 6. If the length is 10 or less, print the word without modification.
 *
 * @section complexity
 * Time Complexity: O(N * L), where N is the number of words and L is the
 * maximum length of a word. This is primarily due to the I/O operations for
 * reading and printing the strings.
 * Space Complexity: O(L), where L is the maximum length of a word, to store
 * one word at a time in memory.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Functional Utility: Decouples C++ streams from C standard streams for faster I/O.
  ios::sync_with_stdio(false);
  
  int n; // Number of words to process
  int strLen; // Length of the current word
  cin >> n;
  string s; // Holds the current word

  // Block Logic: Process n words, one by one.
  for (int i = 0; i < n; ++i) {
    cin >> s;
    strLen = s.length();

    // Pre-condition: 's' contains a word.
    // Invariant: Apply abbreviation rule based on length.
    if (strLen > 10) {
      // Abbreviate the word.
      cout << s[0] << strLen - 2 << s[strLen - 1] << "\n";
    } else {
      // Print the word as is.
      cout << s << "\n";
    }
  }

  return 0;
}
