/**
 * @file 339A.CPP
 * @brief Solution for a competitive programming problem involving sorting a sum.
 * @details The program takes a string representing a sum of the numbers 1, 2, and 3
 * (e.g., "3+2+1") and rearranges it into non-decreasing order (e.g., "1+2+3").
 *
 * @section Algorithm
 * The core algorithm is a specialized version of Counting Sort.
 * 1. Read the input sum as a string.
 * 2. Iterate through the string, counting the occurrences of the characters '1', '2', and '3'.
 * 3. Construct the sorted output string by appending the numbers in order ('1's, then '2's, then '3's),
 *    interspersed with '+' characters. Special care is taken not to place a '+' at the very end.
 *
 * @section Complexity
 * - Time Complexity: O(L), where L is the length of the input string. The process involves a single pass for counting
 *   and another effective pass for printing, making it linear.
 * - Space Complexity: O(1), as the storage required for the counts is constant regardless of the input size.
 */
// ACCEPTED
#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;
  // total: counts the total number of digits (1, 2, or 3).
  // subtotal: tracks the number of digits printed so far to correctly place '+' signs.
  int total = 0, subtotal = 0;
  // c1, c2, c3: counters for the numbers 1, 2, and 3 respectively.
  int c1 = 0, c2 = 0, c3 = 0;
  
  /**
   * Block Logic: Count the occurrences of each number ('1', '2', '3') in the input string.
   * This is the counting phase of the counting sort algorithm. It ignores '+' characters.
   */
  for (char a : s) {
    if (a == '1') {
      c1++;
      total++;
    }
    if (a == '2') {
      c2++;
      total++;
    }
    if (a == '3') {
      c3++;
      total++;
    }
  }

  /**
   * Block Logic: Print the sorted sum by reconstructing it from the counts.
   * This is the output phase of the counting sort.
   */

  // Pre-condition: c1 contains the count of '1's.
  // Invariant: Print '1's separated by '+' until the count of '1's is zero.
  while (c1 != 0) {
    subtotal++;
    if (subtotal != total)
      cout << "1"
           << "+";
    else
      cout << "1";

    c1--;
  }

  // Pre-condition: c2 contains the count of '2's.
  // Invariant: Print '2's separated by '+' until the count of '2's is zero.
  while (c2 != 0) {
    subtotal++;
    if (subtotal != total)
      cout << "2"
           << "+";
    else
      cout << "2";

    c2--;
  }

  // Pre-condition: c3 contains the count of '3's.
  // Invariant: Print '3's separated by '+' until the count of '3's is zero.
  while (c3 != 0) {
    subtotal++;
    if (subtotal != total)
      cout << "3"
           << "+";
    else
      cout << "3";

    c3--;
  }
}
