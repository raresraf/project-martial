/**
 * @file 58A.CPP
 * @brief Solution for the "Chat room" problem (Codeforces 58A).
 * @details This program determines if the word "hello" is present as a subsequence
 * in a given input string. This means the characters 'h', 'e', 'l', 'l', 'o'
 * must appear in order, but not necessarily one after another.
 *
 * Algorithm: Greedy Two-Pointer Subsequence Search
 * The code employs a two-pointer technique. One pointer ('i') iterates through the
 * input string `s`, while a second pointer ('j') tracks the current character in the
 * target string "hello". When a character from the input matches the target character
 * at `ans[j]`, the target pointer `j` is advanced. If `j` successfully reaches the
 * end of "hello", the subsequence has been found.
 *
 * Time Complexity: O(N), where N is the length of the input string, due to the single pass.
 * Space Complexity: O(N) to store the input string.
 */

#include <algorithm>
#include <iostream>

using namespace std;

int main() {

  // s: The input string to be checked.
  string s;
  cin >> s;

  // i: Pointer for the input string 's'.
  // j: Pointer for the target string 'ans'.
  int i, j;
  j = 0;
  
  // The target subsequence to find.
  string ans = "hello";
  
  /**
   * Block Logic: Iterate through the input string 's' to find the subsequence.
   * Pre-condition: 'j' is 0, pointing to the start of the target "hello".
   * Invariant: 'j' holds the length of the prefix of "hello" found so far.
   */
  for (i = 0; i < s.size(); i++) {

    // Check if the current character in the input matches the current character in the target.
    if (s[i] == ans[j]) {

      // If the last character ('o') of "hello" has been found (j==4)...
      if (j == 4) {
        // ...the subsequence is complete. Print "YES" and terminate.
        cout << "YES" << endl;
        return 0;
      }
      // Advance the pointer for the target string.
      j++;
    }
  }
  
  // If the loop completes without finding the full subsequence, print "NO".
  cout << "NO" << endl;

  return 0;
}