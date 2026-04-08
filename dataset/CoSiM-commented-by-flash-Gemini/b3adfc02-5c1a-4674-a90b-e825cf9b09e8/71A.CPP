/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces (Problem 71A).
 * @details This program shortens words that are longer than 10 characters by
 * creating an abbreviation. The abbreviation consists of the first character,
 * the number of characters between the first and last, and the last character.
 *
 * @algorithm
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times to process each word.
 * 3. In each iteration, read a word.
 * 4. Check if the length of the word is greater than 10.
 * 5. If it is, print the abbreviation: the first character, the length minus two,
 *    and the last character.
 * 6. If the length is 10 or less, print the word as is.
 *
 * @note This implementation uses a vector to store words, which is not strictly
 * necessary as each word is processed immediately after being read. The variables
 * `count1` and `count2` are also declared but unused.
 *
 * @time_complexity O(N), where N is the number of words, as each word is processed once.
 * @space_complexity O(L_max * N) due to the vector storage, where L_max is the
 * max word length. This could be O(L_max) if the vector were omitted.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  // Unused variables:
  // int count1 = 1, count2 = 0;
  cin >> n;
  vector<string> s; // Unnecessary storage for words.
  string word;

  /**
   * @block_logic Loop through each word, read it, and apply abbreviation logic.
   */
  for (int i = 0; i < n; i++) {
    cin >> word;
    s.push_back(word); // Storing the word is not needed for the logic.
    if (s[i].size() > 10) {
      // Abbreviate the word.
      cout << s[i][0] << s[i].size() - 2 << s[i][s[i].size() - 1] << endl;
    } else {
      // Print the word as is.
      cout << s[i] << endl;
    }
  }

  return 0;
}
