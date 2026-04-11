/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem: "Way Too Long Words".
 * This program reads an integer 'n' followed by 'n' words. For each word, if its
 * length is greater than 10, it abbreviates the word to its first letter, the
 * number of characters between the first and last letters, and its last letter.
 * Otherwise, it prints the word as is.
 * Algorithm: String length check and conditional abbreviation/printing.
 * Time Complexity: O(L) for each word, where L is the length of the word. Total O(N*L_max).
 * Space Complexity: O(L_max) for storing the word.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * Reads words and applies abbreviation logic based on length.
 */
int main() {
  int n;    // Number of words to process.
  string s; // Stores the current word.
  // Reads the number of words 'n' to be processed.
  cin >> n;
  /**
   * Block Logic: Iterates 'n' times, processing one word in each iteration.
   * Invariant: 'n' decreases with each iteration, moving towards the termination condition.
   * Pre-condition: 'n' is a non-negative integer.
   */
  while (n--) {
    // Reads a word into string 's'.
    cin >> s;
    /**
     * Block Logic: Checks the length of the word and applies abbreviation if necessary.
     * Pre-condition: 's' contains a valid word.
     */
    if (s.length() > 10)
      // If word is too long, prints the abbreviated form: first char, count of middle chars, last char.
      cout << s.at(0) << s.length() - 2 << s.at(s.length() - 1) << endl;
    else
      // Otherwise, prints the word as is.
      cout << s << endl;
  }
  return 0;
}
