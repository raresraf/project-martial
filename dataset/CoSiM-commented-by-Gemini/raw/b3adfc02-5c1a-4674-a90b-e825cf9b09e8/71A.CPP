/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - "Way Too Long Words".
 * @details This program processes a list of words. For each word, it checks if its
 *          length is greater than 10. If it is, the word is replaced by a special
 *          abbreviation. Otherwise, the word is printed as is.
 *
 * Algorithm:
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times:
 *    a. Read a word.
 *    b. If the word's length is strictly greater than 10:
 *       i. Print the first character of the word.
 *      ii. Print the number of characters between the first and last characters (i.e., length - 2).
 *     iii. Print the last character of the word.
 *    c. Otherwise, print the word unchanged.
 *
 * Note: The implementation uses a vector `s` to store the words, which is
 * unnecessary as each word is processed and printed immediately. The variables
 * `count1` and `count2` are also declared but never used.
 *
 * Time Complexity: O(N), where N is the number of words, as each word is processed once.
 * Space Complexity: O(M*N), where M is the max length of a word, due to storing all words in a vector.
 *                   This could be O(M) if the vector were omitted.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // `n` is the number of words to process.
  int n;
  // `count1` and `count2` are declared but are unused variables.
  int count1 = 1, count2 = 0;
  cin >> n;
  // The vector `s` is used to store words, though it is not strictly necessary
  // as each word is processed immediately after being read.
  vector<string> s;
  string word;

  /**
   * @brief Main loop to process `n` words.
   */
  for (int i = 0; i < n; i++) {
    cin >> word;
    s.push_back(word);
    /**
     * @brief Block Logic: Abbreviate the word if its length is greater than 10.
     * The abbreviation consists of the first letter, the count of intermediate
     * letters, and the last letter.
     */
    if (s[i].size() > 10) {
      cout << s[i][0] << s[i].size() - 2 << s[i][s[i].size() - 1] << endl;
      // An empty, commented-out loop.
      // for(int j = 0; j < s[i].size(); j++){}
    } else {
      // If the word is not too long, print it as is.
      cout << s[i] << endl;
    }
  }

  return 0;
}
