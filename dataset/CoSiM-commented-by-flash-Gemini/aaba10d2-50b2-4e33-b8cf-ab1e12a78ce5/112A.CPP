/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two strings.
 * 
 * Logic: Compares two input strings by first normalizing both to lowercase 
 * and then using standard string comparison operators to determine their relative order.
 * 
 * Algorithm: Linear-time state normalization followed by relational comparison.
 * Time Complexity: $O(N)$ where N is the string length.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s1, s2;
  int i, x;
  
  /**
   * Functional Utility: Continuously processes pairs of strings from standard input.
   */
  while (cin >> s1 >> s2) {
    /**
     * Block Logic: Normalizes both strings to lowercase.
     * Invariant: Ensures that comparison is case-insensitive by mapping 
     * all uppercase ASCII characters to their lowercase counterparts.
     */
    for (i = 0; i < s1.size(); i++) {
      s1[i] = (char)tolower(s1[i]);
      s2[i] = (char)tolower(s2[i]);
    }

    /**
     * Functional Utility: Evaluates the relative dictionary order of normalized strings.
     * Output Protocol:
     * - "0"  : Strings are identical.
     * - "-1" : s1 precedes s2.
     * - "1"  : s1 follows s2.
     */
    if (s1 == s2)
      cout << "0" << endl;
    else if (s1 < s2)
      cout << "-1" << endl;
    else if (s1 > s2)
      cout << "1" << endl;
  }
  
  return 0;
}
