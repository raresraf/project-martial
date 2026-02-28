/**
 * @file 58A.CPP
 * @brief Solution for the "Chat room" problem from Codeforces.
 * @details The problem is to determine if a user's typed string contains the word "hello"
 * as a subsequence. This means the characters 'h', 'e', 'l', 'l', 'o' must appear
 * in the input string in that specific order, but not necessarily consecutively.
 * @link https://codeforces.com/problemset/problem/58/A
 *
 * @algorithm Two-Pointer Subsequence Matching
 * This solution uses a two-pointer (or two-index) approach, which is a standard and
 * efficient way to check for a subsequence.
 * 1. Initialize two pointers: `i` for the input string `s`, and `j` for the target
 *    string "hello".
 * 2. Iterate through the input string `s` with pointer `i`.
 * 3. If the character at `s[i]` matches the character at `h[j]`, it means we have found
 *    the next character of our subsequence. We then advance both pointers (`i` and `j`)
 *    to look for the next character of "hello".
 * 4. If the characters do not match, we only advance the pointer `i`, effectively
 *    skipping the character in `s` and continuing to search for the current target `h[j]`.
 * 5. The process continues until we either find all characters of "hello" (`j` reaches 5)
 *    or we run out of characters in the input string `s` (`i` reaches the end).
 * 6. If `j` equals 5 at the end, the subsequence was found.
 *
 * @complexity
 * - Time: O(L), where L is the length of the input string `s`, because we make a single
 *   pass through it.
 * - Space: O(L), to store the input string.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // i: pointer/index for the input string `s`.
  // j: pointer/index for the target string `h`.
  int i = 0, j = 0;
  
  // s: the input string from the user.
  string s;
  // h: the target subsequence we are looking for.
  string h = "hello";
  
  cin >> s;
  int len = s.size();

  /**
   * @brief Iterate through the input string `s` to find the "hello" subsequence.
   * The loop continues as long as we haven't exhausted the input string (`s`)
   * and we haven't yet found all characters of "hello".
   */
  // Invariant: At the start of each iteration, the first `j` characters of "hello"
  // have been found as a subsequence within the first `i` characters of `s`.
  while (i < len && j < 5) {
    // Pre-condition: Check if the current character in `s` matches the target character in `h`.
    if (s[i] == h[j]) {
      // If they match, advance both pointers. We now look for the next character of "hello".
      i++;
      j++;
    } else {
      // If they don't match, only advance the pointer in `s` to continue searching.
      i++;
    }
  }

  /**
   * @brief Check if the entire subsequence was found.
   * If pointer `j` reached 5, it means all characters of "hello" ('h','e','l','l','o')
   * were found in the correct order.
   */
  if (j == 5) {
    // `puts` is a C-style function to print a string followed by a newline.
    puts("YES");
  } else {
    puts("NO");
  }
  
  return 0; // Indicate successful execution.
}
