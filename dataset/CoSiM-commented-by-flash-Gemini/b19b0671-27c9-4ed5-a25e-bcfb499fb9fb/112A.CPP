/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two input strings.
 * 
 * Logic: Normalizes two strings to lowercase and determines their relative 
 * dictionary order. Returns 0 if identical, 1 if the first is greater, 
 * and -1 if the second is greater.
 * 
 * Algorithm: Linear-time state normalization followed by relational comparison.
 * Time Complexity: $O(N)$ where N is the length of the strings.
 * Space Complexity: $O(N)$ for the input string buffers.
 */

#include <cctype>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  string s1, s2;
  
  /**
   * Functional Utility: Continuously processes pairs of strings from standard input.
   */
  while (cin >> s1 >> s2) {
    /**
     * Block Logic: Normalizes both character sequences to lowercase.
     * Invariant: Ensures that subsequent comparison is case-insensitive by mapping 
     * all uppercase ASCII characters to their lowercase equivalents.
     * Note: Loop condition uses comma operator (evaluates s2.size() but both increment).
     */
    for (int i = 0, j = 0; i < s1.size(), j < s2.size(); i++, j++) {
      s1[i] = (char)tolower(s1[i]);
      s2[j] = (char)tolower(s2[j]);
    }

    /**
     * Functional Utility: Evaluates the relative lexicographical order.
     * Result Protocol:
     * - "0"  : Strings are equivalent in a case-insensitive context.
     * - "1"  : s1 follows s2 in dictionary order.
     * - "-1" : s1 precedes s2 in dictionary order.
     */
    if (s1 == s2) {
      cout << "0" << endl;
    } else if (s1 > s2) {
      cout << "1" << endl;
    } else {
      cout << "-1" << endl;
    }
  }
  
  return 0;
}
