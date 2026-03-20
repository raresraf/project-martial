/**
 * @file 58A.CPP
 * @brief Solution for Codeforces problem 58A, "Chat room".
 * @details The program checks if it's possible to form the word "hello" by
 * deleting zero or more characters from a given input string. This is a
 * subsequence checking problem.
 *
 * @section Algorithm
 * This solution uses a state machine, implemented with a series of flags, to
 * verify if "hello" exists as a subsequence in the input string `s`.
 *
 * 1. Five integer flags (`h`, `e`, `l`, `l1`, `o`) are initialized to 0. Each
 *    flag corresponds to a character in "hello".
 * 2. The program iterates through the input string `s` character by character.
 * 3. The `if-else if` chain acts as a state machine:
 *    - It first looks for an 'h'. When found, the `h` flag is set to 1.
 *    - Once `h` is 1, it looks for an 'e'. When found, the `e` flag is set.
 *    - This continues for the first 'l' (flag `l`), the second 'l' (flag `l1`),
 *      and the 'o' (flag `o`).
 * 4. After iterating through the string, if the sum of all flags is 5, it means
 *    all characters of "hello" were found in the correct order. The program
 *    prints "YES". Otherwise, it prints "NO".
 *
 * @subsection Complexity
 * - Time Complexity: O(L), where L is the length of the input string, due to the
 *   single pass through the string.
 * - Space Complexity: O(L), to store the input string.
 */
#include <bits/stdc++.h>
#include <stdlib.h>
#include <string>

using namespace std;

int main() {
  string s;

  /**
   * @brief These integer flags represent the state of the subsequence search.
   * A flag is 0 if the character hasn't been found yet in the correct sequence,
   * and 1 if it has. `l1` is for the second 'l'.
   */
  int h = 0, e = 0, l = 0, l1 = 0, o = 0;

  cin >> s;

  /**
   * @brief This loop iterates through the input string, functioning as a state machine
   * to find the characters of "hello" in order.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == 'h' && h == 0) {
      h++; // Found 'h', advance to next state.
    } else if (s[i] == 'e' && h == 1 && e == 0) {
      e++; // Found 'e' after 'h', advance.
    } else if (s[i] == 'l' && e == 1 && l == 0) {
      l++; // Found first 'l' after 'e', advance.
    } else if (s[i] == 'l' && l == 1 && l1 == 0) {
      l1++; // Found second 'l' after first 'l', advance.
    } else if (s[i] == 'o' && l1 == 1 && o == 0) {
      o++; // Found 'o' after second 'l', final state reached.
    }
  }

  /**
   * @brief If the sum of all flags is 5, it signifies that all characters of
   * "hello" were found sequentially, completing the subsequence.
   */
  if (h + e + l + l1 + o == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
