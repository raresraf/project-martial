/**
 * @file 339A.CPP
 * @brief Solution for the "Helpful Maths" problem (Codeforces 339A).
 * @author competitive-programmer
 *
 * @details This program reorders a sum of numbers (1, 2, and 3), given as a
 * string, into non-decreasing order. For example, an input of "3+2+1"
 * is transformed into "1+2+3".
 *
 * Algorithm: Extract, Sort, and Rebuild
 * 1. Read the input string `s` representing the sum.
 * 2. Create a temporary string `s1` to hold only the digit characters.
 * 3. Iterate through the input string `s`. If a character is not a '+',
 *    append it to `s1`.
 * 4. Use `std::sort` on the `s1` string. This correctly sorts the numbers
 *    as the characters '1', '2', '3' have a corresponding lexicographical order.
 * 5. Rebuild the output string by printing the sorted digits from `s1`,
 *    interspersed with '+' signs.
 *
 * Time Complexity: O(L log L), where L is the number of digits in the sum,
 * dominated by the sorting operation.
 * Space Complexity: O(L) to store the temporary string of digits.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  // Note: j, m, n are declared but not used.
  int i, j, k, l, m, n, o;
  
  // This loop handles multiple inputs, a common competitive programming pattern.
  while (cin >> s) {
    l = s.length();
    // `o` is a flag to check if a '+' was present in the input.
    o = 0;
    // `s1` will store only the digits from the input string.
    string s1;
    
    /**
     * Block Logic: Extract all digit characters from the input string.
     */
    for (i = 0; i < l; i++) {
      if (s[i] != '+') {
        s1 += s[i];
      } else {
        o = 1; // Mark that a '+' was found.
      }
    }
    k = s1.length();
    
    // Sort the string of digits.
    sort(s1.begin(), s1.end());
    
    /**
     * Block Logic: Reconstruct and print the sorted sum.
     */
    if (o == 0) {
      // If there were no '+' signs, just print the single digit.
      for (i = 0; i < k; i++) {
        cout << s1[i];
      }
      cout << endl;
    } else {
      // If there were '+' signs, print the sorted digits separated by '+'.
      for (i = 0; i < k; i++) {
        cout << s1[i];
        if (i < k - 1) {
          cout << '+';
        }
      }
      cout << endl;
    }
  }
  return 0;
}
