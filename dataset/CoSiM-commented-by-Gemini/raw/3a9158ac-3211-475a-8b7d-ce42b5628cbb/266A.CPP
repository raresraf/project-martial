/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem (Codeforces 266A).
 *
 * @details The problem asks for the minimum number of stones to remove from a
 * row of colored stones so that no two adjacent stones have the same color.
 * This solution is written to handle multiple test cases.
 *
 * @algorithm The program iterates through the input string, keeping track of the
 * previously seen character (`last_ch`).
 * 1. For each character in the string, it compares it to `last_ch`.
 * 2. If the current character is the same as the previous one, a counter
 *    for removals is incremented.
 * 3. After the comparison, `last_ch` is updated to the current character
 *    to prepare for the next iteration.
 * This method effectively counts every stone that is part of a consecutive
 * sequence of identical stones, except for the first one in each sequence.
 *
 * @complexity
 * Time Complexity: O(T * N), where T is the number of test cases and N is the
 *                  length of the string. For a single test case, it is O(N).
 * Space Complexity: O(N), to store the input string.
 */
#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int cs, count = 0; // cs: number of stones, count: stones to remove
  string inp;
  char ch, last_ch;
  
  // This loop structure allows for multiple test cases, though the problem has only one.
  while (cin >> cs) {
    count = 0;
    cin >> inp;
    /**
     * @brief Block Logic: Iterates through the stones to count removals.
     * @invariant `count` holds the number of removals required for the prefix
     * `inp[0...i-1]`. `last_ch` holds the color of the stone at `i-1`.
     */
    for (int i = 0; i < inp.size(); i++) {
      // Special handling for the first stone to initialize `last_ch`.
      if (i == 0) {
        last_ch = inp[i];
        continue;
      }
      ch = inp[i];
      if (ch == last_ch) {
        count++;
      }
      last_ch = ch;
    }
    cout << count << endl;
  }
}
