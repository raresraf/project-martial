/**
 * @file 58A.CPP
 * @brief An incorrect solution attempt for the "Chat room" problem (Codeforces 58A).
 *
 * @algorithm
 * The problem is to determine if the word "hello" appears as a subsequence in a
 * given string. For example, "ahhellllloou" contains "hello".
 *
 * This code attempts to solve the problem by iterating through both the input
 * string and the target string "hello". However, the implemented logic is
 * fundamentally flawed.
 *
 * @warning
 * FLAWED IMPLEMENTATION: This code does not correctly solve the problem.
 * 1.  **Logic Error**: The main loop's logic for matching the subsequence is
 *     incorrect. The `else { i--; j++; }` block causes the algorithm to behave
 *     unpredictably and not like a proper subsequence check.
 * 2.  **Out-of-Bounds Access**: The loop `for (i = 0; i < l; i++)` uses `i` to index
 *     `str2` ("hello"), which has a fixed size of 5 (plus null terminator). If
 *     the input string `str` has a length `l` greater than 5, this will read
 *     past the bounds of `str2`, leading to undefined behavior.
 */
#include <bits/stdc++.h>
#include <cstring>
using namespace std;

int main() {
  // `str` for input, `str2` for the target "hello". C-style strings are used.
  char str[101], str2[10];
  // `j` is used as an index for `str`, `count` tracks matched characters.
  // `flag` is set to 1 if "hello" is supposedly found.
  // `k`, `m`, `n` are unused variables.
  int i, j = 0, k, l, m, n, count = 0, flag = 0;

  cin >> str;
  l = strlen(str);

  strcpy(str2, "hello");

  /**
   * @brief This loop is intended to find the "hello" subsequence but is flawed.
   * The loop iterates up to `l` (length of input), but `i` is used to index
   * `str2` ("hello"), causing an out-of-bounds read if `l > 5`.
   * The logic inside does not correctly perform a subsequence search.
   */
  for (i = 0; i < l; i++) {
    // Compares a character from "hello" with a character from the input string.
    if (str2[i] == str[j]) {
      j++;
      count++;

      if (count == 5) {
        flag = 1;
        break;
      }
    } else {
      // FLAWED LOGIC: If there's no match, the code decrements the index `i` for
      // "hello" while advancing the index `j` for the input string. This does
      // not correctly find a subsequence.
      i--;
      j++;
    }

    // Breaks if the end of the input string is reached.
    if (j == l)
      break;
  }

  // Final output based on whether the `flag` was set.
  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
