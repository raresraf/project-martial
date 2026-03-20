/**
 * @file 58A.CPP
 * @brief Solution for the "Chat room" problem (Codeforces 58A).
 * @details This program determines if the word "hello" is a subsequence of a
 * given input string. A subsequence can be formed by deleting zero or more
 * characters from the original string.
 *
 * @section algorithm Algorithm
 * A greedy approach is used. The code maintains a pointer (`k`) to the current
 * character being sought in the target word "hello". It iterates through the
 * input string, and whenever a character matches the target character, the pointer
 * is advanced. If the pointer successfully traverses the entire target word,
 * the subsequence exists.
 *
 * @section complexity Complexity
 * - Time Complexity: O(S), where S is the length of the input string.
 * - Space Complexity: O(S) for storing the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string str, tmp = "hello";
  int k = 0;
  cin >> str;
  /**
   * @brief Main loop to find the subsequence.
   * Iterates through the input string `str` to find characters of `tmp` in order.
   */
  for (int i = 0; i < str.size(); i++) {
    // Optimization: If all characters of "hello" have been found, stop early.
    if (k == tmp.size())
      break;
    // If the current character in the input matches the target character...
    if (tmp[k] == str[i])
      // ...advance the pointer to look for the next character in "hello".
      k++;
  }

  // After checking the whole string, if k has reached the length of "hello",
  // it means the full subsequence was found.
  if (k == tmp.size()) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
