/**
 * @file 58A.CPP
 * @brief Solution to the "Chat room" problem (Codeforces 58A).
 *
 * @details This program determines if a given string contains "hello" as a subsequence.
 * A subsequence can be formed by deleting zero or more characters from the original string
 * without changing the order of the remaining characters.
 *
 * @algorithm The program employs a greedy two-pointer strategy. One pointer (`i`)
 * iterates through the input string `a`, while a second pointer (`j`) tracks the
 * current position in the target string "hello". When the character `a[i]` matches
 * the character `b[j]`, it signifies that the next character of the subsequence has
 * been found, so the `j` pointer is advanced. If the `j` pointer successfully
 * traverses all 5 characters of "hello", the program confirms the subsequence exists.
 * A counter `pas` tracks the number of matched characters.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, due to the single pass.
 * Space Complexity: O(L), to store the input string `a`.
 */
#include <iostream>
using namespace std;
// Global variables to hold the input string, target string, and state for the search.
string a, b = "hello";
int j = 0, pas = 0;
int main() {
  cin >> a;
  /**
   * @brief Block Logic: Iterates through the input string to find the "hello" subsequence.
   * @invariant `j` is the index of the next character to find in "hello". `pas` is the
   * number of characters from "hello" found so far in the correct order.
   */
  for (int i = 0; i < a.size(); i++) {
    // If the current character in the input matches the target character in "hello"...
    if (a[i] == b[j]) {
      // ...advance to the next character in "hello" and increment the found-character count.
      j++;
      pas++;

      // Early exit if all 5 characters of "hello" have been found.
      if (pas == 5) {
        break;
      }
    }
  }
  
  // If all 5 characters were found, "hello" is a subsequence.
  if (pas == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
