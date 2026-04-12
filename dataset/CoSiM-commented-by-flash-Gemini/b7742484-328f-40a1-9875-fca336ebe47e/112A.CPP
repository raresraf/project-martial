/**
 * @file 112A.CPP
 * @brief Case-insensitive lexicographical comparison of two input strings.
 * 
 * Logic: Compares two strings character-by-character while ignoring case differences. 
 * Both strings are normalized to lowercase before applying relational operators. 
 * Returns 1 if the first string is greater, -1 if the second is greater, or 0 if identical.
 * 
 * Algorithm: Sequential state normalization followed by relational evaluation.
 * Time Complexity: $O(N)$ where N is the length of the strings.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  string a, b;
  // Functional Utility: Reads two character sequences from standard input.
  if (!(cin >> a >> b)) return 0;

  /**
   * Block Logic: Normalizes the first string to lowercase.
   * Invariant: Maps uppercase ASCII characters to their lowercase equivalents.
   */
  for (int i = 0; i < (int)a.size(); i++)
    a[i] = (char)tolower(a[i]);

  /**
   * Block Logic: Normalizes the second string to lowercase.
   */
  for (int i = 0; i < (int)b.size(); i++)
    b[i] = (char)tolower(b[i]);

  /**
   * Functional Utility: Lexicographical order evaluation.
   * Result Protocol:
   * - "1"  : 'a' follows 'b' in dictionary order.
   * - "0"  : strings are equivalent in a case-insensitive context.
   * - "-1" : 'a' precedes 'b' in dictionary order.
   */
  if (a > b)
    cout << 1;
  else if (a == b)
    cout << 0;
  else
    cout << -1;
    
  return 0;
}
