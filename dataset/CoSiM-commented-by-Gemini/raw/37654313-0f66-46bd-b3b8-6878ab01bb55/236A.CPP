/**
 * @file 236A.CPP
 * @brief Determines an outcome based on the parity of distinct characters in a string.
 *
 * @algorithm Count distinct characters via sorting. The string is sorted to group
 *            identical characters. A single pass then counts unique characters by
 *            detecting changes between adjacent elements.
 * @complexity
 * Time Complexity: O(N log N), dominated by the sort operation on the string of length N.
 * Space Complexity: O(N) for storing the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  char s[100];
  cin >> s;
  int i, j, c = 0;
  int l = strlen(s);
  
  // Pre-computation: Sort the character array to group identical characters together.
  // This is a prerequisite for the linear-time counting of distinct elements.
  sort(s, s + l);

  /**
   * @brief Block Logic: Iterates through the sorted array to count unique characters.
   * @invariant `c` stores the number of unique characters discovered in the scanned prefix of the string.
   * A character is considered unique at the last of its contiguous block.
   */
  for (i = 0; i < l; i++) {
    if (s[i] != s[i + 1]) {
      c++;
    }
  }

  /**
   * @brief Block Logic: Final decision output based on the parity of the unique character count.
   * An even count results in one output, an odd count in another.
   */
  if (c % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else
    cout << "IGNORE HIM!";
}
