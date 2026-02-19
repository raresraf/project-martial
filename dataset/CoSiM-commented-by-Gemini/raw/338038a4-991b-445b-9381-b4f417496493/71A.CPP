/**
 * @file 71A.CPP
 * @brief An implementation for the "Way Too Long Words" problem.
 * @details This program abbreviates words longer than 10 characters.
 * @note This implementation has common bugs related to C++ stream input. The loop
 * runs `n+1` times, and the first `getline` call after `cin >> n` reads an empty
 * string due to a leftover newline character in the input buffer.
 *
 * @algorithm The program reads an integer `n`. It then attempts to loop `n` times to process
 * `n` words. Inside the loop, it reads a full line into a string. If the string's length
 * is greater than 10, it prints an abbreviation (first letter, count of middle
 * characters, last letter). Otherwise, it prints the original string.
 *
 * @time_complexity O(L_total), where L_total is the sum of the lengths of all lines read.
 * @space_complexity O(L_max), where L_max is the maximum length of any single line.
 */
#include <bits/stdc++.h>
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  cin >> n;
  string str;
  /**
   * Block Logic: Loop to read and process n+1 lines from input.
   * @note This loop runs n+1 times due to `i <= n`. The first iteration with `i=0`
   * will consume the newline character left by `cin >> n`, resulting in an empty `str`.
   * The subsequent n iterations will read the actual input words.
   */
  for (int i = 0; i <= n; i++) {
    getline(cin, str);

    // For the first iteration (i=0), `str` will be empty and this prints a blank line.
    if (str.size() > 10) {
      cout << str[0] << str.size() - 2 << str[str.size() - 1] << endl;

    } else {
      cout << str << endl;
    }
  }
}
