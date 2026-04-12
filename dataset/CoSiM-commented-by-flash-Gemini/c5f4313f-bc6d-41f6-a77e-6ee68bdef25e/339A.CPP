/**
 * @file 339A.CPP
 * @brief Rearranges an arithmetic sum string by sorting its constituent numbers in ascending order.
 *   This program processes a string representing a sum (e.g., "1+5+3") and outputs the sum
 *   with its numbers sorted (e.g., "1+3+5").
 * * Algorithm: Extraction, Sorting, and Reconstruction.
 *   1. Numbers are extracted from the input string.
 *   2. The extracted numbers are sorted in ascending order.
 *   3. The sorted numbers are then printed, interleaved with '+' signs.
 * Time Complexity: O(L + (L/2)log(L/2)) where L is the length of the string.
 *   The dominant factor is the sorting of numbers, which is proportional to the number of numbers (L/2)
 *   multiplied by the logarithm of the number of numbers.
 * Space Complexity: O(L) for storing the extracted numbers.
 */

#include <bits/stdc++.h>
using namespace std;

#define ll long long

/**
 * @brief Main function to process the arithmetic sum string.
 * Functional Utility: Handles input, extracts and sorts numbers, and prints the sorted sum.
 */
int main() {
  bool flag = false;
  string s;
  cin >> s;

  // Calculates the maximum possible number of integers in the string.
  // Example: for "1+2+3", length is 5, (5/2)+1 = 3 integers.
  int n[s.length() / 2 + 1];

  /**
   * Block Logic: Extracts numbers (digits) from the input string.
   * Invariant: After this loop, 'n' array contains all the numbers from 's'.
   */
  for (int i = 0; i < s.length(); i += 2) {
    // Inline: Converts a character digit to its integer value.
    n[i / 2] = s[i];
  }

  // Algorithm: Sorts the extracted numbers in ascending order.
  sort(n, n + (s.length() / 2 + 1));

  /**
   * Block Logic: Reconstructs and prints the arithmetic sum with sorted numbers.
   * Pre-condition: 'n' array contains sorted numbers.
   * Invariant: The output string represents the sorted sum.
   */
  for (int i = 0; i < s.length() / 2 + 1; i++) {
    // Condition: Prints '+' before a number if it's not the first number.
    if (flag)
      cout << "+";
    flag = true;
    // Inline: Converts the ASCII value back to the actual digit value.
    cout << n[i] - 48;
  }
}
