/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem that abbreviates long words.
 * This program reads a list of words and, for each word longer than 10 characters,
 * replaces it with a shortened version consisting of the first letter, the count of
 * intermediate letters, and the last letter. Words 10 characters or shorter are
 * printed as is.
 *
 * Algorithm: String processing with conditional abbreviation.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L) for storing the current word.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function of the program.
 * Reads an integer `n` representing the number of words, then iterates `n` times.
 * In each iteration, it reads a string `s` and applies a conditional abbreviation logic.
 * @return `0` upon successful execution.
 */
int main() {
  int n;
  cin >> n;
  /**
   * @brief Loop through each word to apply abbreviation logic.
   * Invariant: `i` ranges from 0 to `n-1`, processing each input word.
   */
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    int l = s.size();
    /**
     * @brief Conditional logic for word abbreviation.
     * If the word length `l` is 10 or less, print the word as is.
     * Otherwise, print an abbreviated version: first character + (length - 2) + last character.
     * This logic is applied to each word in the input.
     */
    if (l <= 10) {
      cout << s << endl;
    } else {
      cout << s.substr(0, 1) << l - 2 << s.substr(l - 1, l - 1) << endl;
      ;
    }
  }

  return 0;
}
