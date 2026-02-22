/**
 * @file 58A.CPP
 * @brief A correct and efficient solution to the "Chat room" problem (Codeforces 58A).
 *
 * @algorithm
 * The problem is to determine if the word "hello" appears as a subsequence in a
 * given string. A subsequence is a sequence that can be derived from another
 * sequence by deleting some or no elements without changing the order of the
 * remaining elements.
 *
 * This solution uses the classic two-pointer algorithm for finding a subsequence:
 * 1. Initialize a pointer `j` to the beginning of the target string "hello".
 * 2. Initialize a pointer `i` to the beginning of the input string `s`.
 * 3. Iterate through the input string `s` with pointer `i`.
 * 4. If the character at `s[i]` matches the character at `t[j]`, it means we've
 *    found the next character of the subsequence, so we advance both pointers (`i` and `j`).
 * 5. If they do not match, we only advance the pointer `i`, looking for a match
 *    for the current character of `t` further in `s`.
 * 6. The loop continues until either the end of the input string `s` is reached
 *    or the end of the target "hello" is reached.
 * 7. If the pointer `j` has successfully traversed all 5 characters of "hello",
 *    the subsequence exists.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the length of the input string, because we
 *   make a single pass through it.
 * - Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s; // The input string from the user.
  string t; // The target subsequence, "hello".
  cin >> s;
  t = "hello";
  /**
   * @brief A simple optimization: if the input string is shorter than "hello",
   * it's impossible for it to be a subsequence.
   */
  if (s.length() < 5) {
    cout << "NO" << endl;
    return 0;
  }

  int i = 0; // Pointer for the input string `s`.
  int j = 0; // Pointer for the target string `t`.
  /**
   * @brief The two-pointer algorithm loop.
   * Continues as long as we haven't exhausted the input string or found the
   * complete "hello" subsequence.
   */
  while (i < s.length() && j < 5) {
    // If we find a character matching the current target character...
    if (s[i] == t[j]) {
      // ...we advance both pointers.
      i++;
      j++;
    } else {
      // Otherwise, we only advance the input string pointer to keep searching.
      i++;
    }
  }
  /**
   * @brief Check if the entire "hello" subsequence was found.
   * If `j` is 5, it means we have successfully matched all 5 characters
   * of "hello" in order.
   */
  if (j >= 5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}