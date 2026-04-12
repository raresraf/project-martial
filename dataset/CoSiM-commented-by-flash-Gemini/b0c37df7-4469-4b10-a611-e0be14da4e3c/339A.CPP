/**
 * @file 339A.CPP
 * @brief Logic for reordering a numeric summation expression into non-decreasing order.
 * 
 * Logic: Transforms a summation string (e.g., "3+2+1") into its canonical sorted form 
 * ("1+2+3"). Extracts numeric operands, performs a sort, and re-inserts the '+' 
 * delimiters during output.
 * 
 * Algorithm: Operand extraction followed by character sorting.
 * Time Complexity: $O(N \log N)$ where N is the number of operands.
 * Space Complexity: $O(N)$ for the intermediate operand buffer.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  // Functional Utility: Reads the raw summation expression.
  if (!(cin >> s)) return 0;
  
  // Data Structure: Buffer to hold numeric operands, skipping the '+' delimiters.
  char a[s.length()]; 
  int operand_count = 0;

  /**
   * Block Logic: Operand extraction.
   * Invariant: Scans the string at even indices to collect digits, bypassing the '+' at odd indices.
   */
  for (int i = 0; i < (int)s.length(); i = i + 2) {
    a[operand_count++] = s[i];
  }

  // Logic: Orders the operands ascending to satisfy the "Helpful Maths" constraint.
  sort(a, a + operand_count);

  /**
   * Block Logic: Formatted reassembly.
   * Logic: Interleaves sorted operands with '+' delimiters for valid expression output.
   */
  for (int i = 0; i < operand_count; i++) {
    cout << a[i];
    // Condition: Appends the delimiter only between operands.
    if (i != operand_count - 1)
      cout << "+";
  }
  
  return 0;
}
