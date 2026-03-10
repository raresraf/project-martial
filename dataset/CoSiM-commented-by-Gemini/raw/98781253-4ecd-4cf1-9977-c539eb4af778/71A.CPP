/**
 * @file 71A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Way Too Long Words").
 * This program processes a list of words and shortens any word that is longer
 * than 10 characters. The shortened word consists of its first character,
 * the number of characters between the first and last, and the last character.
 *
 * @algorithm The program iterates through each word, checks its length, and if
 * it exceeds 10, it constructs and replaces the original string with its
 * abbreviated form before printing.
 *
 * @time_complexity O(N * L), where N is the number of words and L is the maximum
 * length of a word. This is primarily dominated by the string I/O operations.
 * @space_complexity O(L) to store the longest word.
 */
//
// Created by Taewoo Kim on 2/7/2017.
//

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n; // Number of words to process
  string s;
  cin >> n;

  /**
   * @block
   * @brief Processes each word according to the abbreviation rule.
   */
  for (int i = 0; i < n; i++) {
    cin >> s;
    // Pre-condition: Check if the word's length is greater than 10.
    if (s.size() > 10)
      // If it is, construct the abbreviation and reassign it to the string `s`.
      s = s[0] + to_string(s.size() - 2) + s[s.size() - 1];
    cout << s << endl;
  }

  return 0;
}