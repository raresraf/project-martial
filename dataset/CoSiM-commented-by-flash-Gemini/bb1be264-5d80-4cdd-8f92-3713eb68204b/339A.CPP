/**
 * @file 339A.CPP
 * @brief Logic for reordering a numeric summation expression into non-decreasing order.
 * 
 * Logic: Extracts numeric digits from an infix summation string (e.g., "3+2+1") 
 * and reassembles them in a canonical sorted form ("1+2+3"). Processes the string 
 * by isolating operands, performing an algebraic sort, and re-inserting delimiters.
 * 
 * Algorithm: Operand extraction followed by lexicographical sorting and formatted output.
 * Time Complexity: $O(N \log N)$ where N is the number of operands.
 * Space Complexity: $O(N)$ for the intermediate operand buffer.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

// Functional Utility: Standardized competitive programming macros for I/O and loops.
#define MX 100000
#define fs first
#define sec second
#define TC                                                                     \
  int T;                                                                       \
  cin >> T;                                                                    \
  while (T--)
#define loop(i, a, c) for (int i = 0; i < a; i++)
#define loop1(i, a, c) for (int i = 1; i <= a; i++)
#define pb(a) push_back(a)
#define all(a) (a).begin(), (a).end()
#define mp(x, y) make_pair((x), (y))
#define ll long long int

using namespace std;

int main() {
  string s;
  // Functional Utility: Reads the raw summation expression.
  if (!(cin >> s)) return 0;
  
  string operands = "";
  /**
   * Block Logic: Operand extraction.
   * Invariant: Filters out the '+' delimiters to isolate individual numeric values.
   */
  for (int i = 0; i < (int)s.size(); i++) {
    if (s[i] != '+') {
      operands += s[i];
    }
  }

  // Logic: Reorders the operands ascending to satisfy the "Helpful Maths" ordering rule.
  sort(operands.begin(), operands.end());

  /**
   * Block Logic: Formatted reassembly.
   * Logic: Interleaves the sorted operands with '+' separators.
   * Note: The final operand is emitted separately to avoid a trailing delimiter.
   */
  for (int i = 0; i < (int)operands.size() - 1; i++) {
    cout << operands[i] << "+";
  }
  
  if (!operands.empty()) {
    cout << operands[operands.size() - 1];
  }
  
  return 0;
}
/**
 * Note: Source contains several unused macros typical of competitive programming templates.
 */
