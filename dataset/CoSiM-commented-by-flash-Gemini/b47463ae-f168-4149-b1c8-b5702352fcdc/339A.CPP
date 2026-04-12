/**
 * @file 339A.CPP
 * @brief Logic for reordering a numeric summation expression into non-decreasing order.
 * 
 * Logic: Extracts numeric digits from an infix summation string (e.g., "3+2+1") 
 * and reassembles them in a canonical sorted form ("1+2+3"). Uses frequency counting 
 * for '1's, '2's, and '3's to reconstruct the resulting sequence.
 * 
 * Algorithm: Frequency counting followed by string re-concatenation.
 * Time Complexity: $O(N)$ where N is the string length.
 * Space Complexity: $O(N)$ for result string storage.
 */

#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  string s;
  // Functional Utility: Reads the entire summation expression into a buffer.
  if (!getline(cin, s)) return 0;
  
  // Counters for individual numeric categories.
  int c1 = 0, c2 = 0, c3 = 0;

  /**
   * Functional Utility: Edge case handling for expressions with a single operand.
   */
  if (s.length() < 3) {
    if (!s.empty()) cout << s[0];
    return 0;
  }

  /**
   * Block Logic: Frequency analysis of operands.
   * Invariant: Increments category counters for each identified digit.
   */
  for (int i = 0; i < (int)s.length(); i++) {
    if (s[i] == '1')
      c1++;
    if (s[i] == '2')
      c2++;
    if (s[i] == '3')
      c3++;
  }

  /**
   * Block Logic: Reconstructs the expression from counted categories.
   * Case 1: Start with '1's.
   * Case 2: Start with '2's if no '1's exist.
   * Case 3: Start with '3's if only '3's exist.
   */
  if (c1 > 0) {
    string res = "1";
    for (int i = 1; i < c1; i++)
      res = res + "+1";
    for (int i = 0; i < c2; i++)
      res = res + "+2";
    for (int i = 0; i < c3; i++)
      res = res + "+3";
    cout << res;
  } else if (c2 > 0) {
    string res = "2";
    for (int i = 1; i < c2; i++)
      res = res + "+2";
    for (int i = 0; i < c3; i++)
      res = res + "+3";
    cout << res;
  } else if (c3 > 0) {
    string res = "3";
    for (int i = 1; i < c3; i++)
      res = res + "+3";
    cout << res;
  }

  return 0;
}
