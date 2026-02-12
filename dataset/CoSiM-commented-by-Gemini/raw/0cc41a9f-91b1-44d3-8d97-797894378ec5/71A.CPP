/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @details This program processes a given number of words. For each word, it checks if its
 * length exceeds 10 characters. If it does, the word is abbreviated by printing its first character,
 * the number of characters between the first and last (i.e., length - 2), and its last character.
 * Otherwise, the word is printed in its original form.
 *
 * @algorithm String length checking and conditional printing.
 * @complexity
 * Time complexity: O(T * L), where T is the number of test cases and L is the maximum length of a word.
 * Space complexity: O(L) to store the word string in each iteration.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  // t: the number of test cases (words) to process.
  int t;
  cin >> t;
  /**
   * @brief Main loop to process each word.
   * @invariant Iterates `t` times.
   */
  while (t--) {
    string s;
    cin >> s;
    int n = s.length();
    
    /**
     * @brief Check if the word's length is greater than 10.
     */
    if (n > 10) {
      // f: first character, l: last character.
      char f = s[0];
      char l = s[n - 1];
      // Print the abbreviation.
      cout << f << n - 2 << l << endl;
    } else {
      // Print the original word if it's not too long.
      cout << s << endl;
    }
  }

  return 0;
}