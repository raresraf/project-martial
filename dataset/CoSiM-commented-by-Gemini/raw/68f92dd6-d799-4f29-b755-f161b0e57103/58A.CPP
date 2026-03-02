#include <iostream>
#include <string>
using namespace std;

/**
 * @file 58A.CPP
 * @brief Solution to the "Chat room" problem from Codeforces (Problem 58A).
 * @details This program checks if the word "hello" can be formed as a subsequence
 * from the characters of a given input string.
 *
 * Algorithm:
 * This solution uses a two-pointer/index-based approach, which is a standard
 * way to find a subsequence.
 * 1. The target word "hello" is defined.
 * 2. An index `j` is used to track the current character being sought from "hello".
 * 3. The code iterates through the input string with index `i`.
 * 4. If the character at `s[i]` matches the character at `hello[j]`, it means the
 *    next part of the subsequence has been found, so `j` is incremented.
 * 5. After the loop, if `j` has reached 5, it means all characters of "hello" were
 *    found in the correct order.
 *
 * Time Complexity: O(L), where L is the length of the input string, due to the single
 * pass through it.
 * Space Complexity: O(L) to store the input string.
 */
int main() {
  string s;
  cin >> s;
  // `a` is the target subsequence "hello".
  char a[] = {'h', 'e', 'l', 'l', 'o'};
  // `j` is an index for the target array `a`.
  int j = 0;
  /**
   * @brief Iterate through the input string `s` to find the subsequence.
   */
  for (int i = 0; i < s.length(); i++) {
    // If the current character in `s` matches the current target character in `a`...
    if (s[i] == a[j]) {
      // ...move to the next character in the target.
      j++;
    }
  }
  // After the loop, `j` will be 5 if the entire "hello" subsequence was found.
  if (j < 5)
    cout << "NO";
  else {
    cout << "YES";
  }
  return 0;
}