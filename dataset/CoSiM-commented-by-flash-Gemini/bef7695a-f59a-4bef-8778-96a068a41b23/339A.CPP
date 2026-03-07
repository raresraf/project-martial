/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" competitive programming problem (Codeforces 339A).
 * The problem requires sorting numbers in a sum that is given as a string.
 * The input string contains numbers '1', '2', or '3' separated by '+' signs.
 * The task is to rearrange the numbers in non-decreasing order and print the resulting sum.
 * Algorithm: Bubble sort on characters at even indices (representing numbers).
 * Time Complexity: O(L^2), where L is the length of the input string.
 * Space Complexity: O(L) for storing the string.
 */
#include <iostream> // For input/output operations.
#include <string>   // For string manipulation.
#include <algorithm> // Required for `std::swap`.
using namespace std;

// Declares 's' as a global string to store the input sum.
string s;

int main() {
  // Reads the input sum string from standard input.
  cin >> s;
  
  /**
   * Block Logic: Implements a form of bubble sort to sort the numbers in the sum.
   * Numbers are located at even indices (0, 2, 4, ...).
   * Invariant (outer loop): After each iteration of the outer loop, the character at `s[i]` is in its correct sorted position relative to other characters processed so far.
   * Invariant (inner loop): The inner loop compares `s[i]` with subsequent characters `s[j]` and swaps them if `s[i]` is greater than `s[j]`.
   * Pre-condition: The input string `s` contains digits '1', '2', or '3' at even indices, and '+' at odd indices.
   * Post-condition: The digits in `s` at even indices are sorted in non-decreasing order.
   */
  for (int i = 0; i < s.size(); i += 2) { // Outer loop iterates through each number (even indices).
    for (int j = i + 2; j < s.size(); j += 2) { // Inner loop compares with subsequent numbers.
      /**
       * Conditional Logic: If the current number `s[i]` is greater than a subsequent number `s[j]`.
       * The comparison is done directly on character values, which works for '1', '2', '3'.
       */
      if (s[i] > s[j])
        swap(s[i], s[j]); // Swaps the characters to put them in non-decreasing order.
    }
  }
  // Prints the sorted sum string to standard output.
  cout << s;
  
  return 0; // Indicates successful program execution.
}
