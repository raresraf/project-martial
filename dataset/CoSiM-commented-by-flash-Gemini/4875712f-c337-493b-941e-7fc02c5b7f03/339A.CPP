/**
 * @4875712f-c337-493b-941e-7fc02c5b7f03/339A.CPP
 * @brief This program reorders an arithmetic sum string, ensuring the numbers are in non-decreasing order.
 * Algorithm: Counting sort (or frequency array) to store and then reconstruct the sorted sum.
 * Time Complexity: O(L) where L is the length of the input string, as it involves two linear passes.
 * Space Complexity: O(C) where C is the range of possible digits (e.g., 1-3), for the frequency array.
 */
#include <bits/stdc++.h>
#include <ctype.h>
using namespace std;
int main() {
  string s; // Declare a string variable 's' to store the input sum expression.
  // Functional Utility: Read the input sum expression string.
  cin >> s;
  int len = s.size(); // Store the length of the input string for iteration.
  // Declare an array 'a' of size 4 (indices 0-3) to store the frequency of digits '1', '2', '3'.
  // Index 0 is unused, indices 1, 2, 3 correspond to digits 1, 2, 3 respectively.
  int a[4] = {0};
  int count = 0; // This variable 'count' is declared but not used in the provided code.

  // Block Logic: Parse the input string to count the occurrences of each digit (1, 2, 3).
  // Iterate through the string, skipping '+' characters.
  for (int i = 0; i < len; i++) {
    if (s[i] != '+') {
      // Convert character digit to integer and increment its frequency in array 'a'.
      // 's[i] - '0'' converts '1' to 1, '2' to 2, etc.
      a[(s[i] - '0')]++;
    }
  }

  int k = 1; // Initialize 'k' as a pointer to the current digit (starting from 1) to be printed.
  // Block Logic: Reconstruct and print the sum expression with digits in non-decreasing order.
  // Iterate through the original string length to determine positions for digits and '+' signs.
  for (int i = 0; i < len; i++) {
    if (i % 2 == 0) { // Even indices are for digits.
      // Pre-condition: Find the smallest digit (1, 2, or 3) that still has occurrences.
      while (a[k] == 0)
        k++; // Increment 'k' until a digit with a positive count is found.

      cout << k;  // Print the current smallest available digit.
      a[k]--;     // Decrement the count of the printed digit.
    } else {      // Odd indices are for '+' signs.
      cout << "+"; // Print the '+' separator.
    }
  }

  return 0; // Indicate successful program termination.
}
