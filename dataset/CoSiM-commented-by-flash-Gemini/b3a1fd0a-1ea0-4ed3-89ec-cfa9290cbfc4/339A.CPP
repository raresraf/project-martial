/**
 * @file 339A.CPP
 * @brief This competitive programming solution takes a simple sum expression
 *        (e.g., "3+1+2") consisting only of numbers '1', '2', and '3' separated by
 *        '+' signs. It reorders the numbers in non-decreasing order and then
 *        prints the reordered expression. This specific implementation uses
 *        a direct sorting approach on the input string.
 * Algorithm: In-place Bubble Sort on String Characters. The code directly manipulates
 *            the input string. It iterates through the string with nested loops,
 *            comparing only the digit characters (skipping the '+' signs) and
 *            swapping them to achieve a non-decreasing order.
 * Time Complexity: O(N^2), where N is the number of digits in the string. This is due
 *                  to the nested loops performing comparisons and swaps. If L is the
 *                  length of the string, N is approximately L/2, so it's O(L^2).
 * Space Complexity: O(L), for storing the input string. The sorting is performed
 *                   in-place, requiring no additional data structures proportional
 *                   to the input size.
 */

#include <bits/stdc++.h> // Includes common standard library headers, including iostream and string.
using namespace std;

/**
 * @brief Main function of the program. Reads the expression, sorts its numeric parts, and prints the result.
 * Functional Utility: Serves as the entry point for the "Helpful Maths" problem, performing the core logic
 *                     of sorting digits within an arithmetic expression.
 */
int main() {
  string s;   // Declares a string to hold the input arithmetic expression.
  cin >> s;   // Reads the entire expression string from standard input.

  // Block Logic: Outer loop for the bubble sort algorithm. It iterates through the string,
  //              focusing on the digit characters at even indices (0, 2, 4, ...).
  // Invariant: After each pass of the outer loop, the digit at index 'i' (and possibly earlier digits)
  //            is in its correct sorted position relative to the elements already processed.
  for (int i = 0; i < s.size(); i += 2) {
    // Block Logic: Inner loop for the bubble sort algorithm. Compares the digit at 'i' with
    //              previous digits at even indices (0, 2, ..., j).
    // Invariant: The largest unsorted digit "bubbles up" to its correct position.
    for (int j = 0; j < i; j += 2) {
      // Block Logic: Compares two digit characters and swaps them if they are out of order.
      // Functional Utility: Ensures that smaller digits move towards the beginning of the string.
      if (s[i] < s[j]) {
        swap(s[i], s[j]); // Swaps the characters (digits) directly in the string.
      }
    }
  }
  cout << s; // Prints the modified string with sorted digits.
  return 0;  // Indicates successful program termination.
}
