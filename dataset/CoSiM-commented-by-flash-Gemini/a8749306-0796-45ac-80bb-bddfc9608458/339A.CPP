/**
 * @file 339A.CPP
 * @brief Logic for reordering a numeric summation expression into non-decreasing order.
 * 
 * Algorithm: Multi-pass categorical extraction.
 * Logic: Extracts and groups identical numeric entities ('1', '2', '3') from an 
 * infix summation string, then reassembles them into a canonical sorted form.
 * 
 * Time Complexity: $O(N)$ where N is the length of the expression.
 * Space Complexity: $O(N)$ for the result string construction.
 */

#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string expression, result;
  // Functional Utility: Reads the raw summation sequence (e.g., "3+2+1").
  cin >> expression;
  int length = expression.length();

  /**
   * Block Logic: Categorical extraction of entity '1'.
   * Invariant: Scans indices i=0, 2, 4... to bypass '+' delimiters.
   */
  for (int i = 0; i < length; i += 2) {
    if (expression[i] == '1')
      result.append("1+");
  }

  /**
   * Block Logic: Categorical extraction of entity '2'.
   */
  for (int i = 0; i < length; i += 2) {
    if (expression[i] == '2')
      result.append("2+");
  }

  /**
   * Block Logic: Categorical extraction of entity '3'.
   */
  for (int i = 0; i < length; i += 2) {
    if (expression[i] == '3')
      result.append("3+");
  }

  /**
   * Functional Utility: Normalizes the resulting sequence by removing the trailing 
   * '+' operator.
   */
  if (!result.empty()) {
    result.pop_back();
  }
  
  // Output the reordered summation sequence.
  cout << result;

  return 0;
}
