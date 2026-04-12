/**
 * @file 339A.CPP
 * @brief Rearranges an arithmetic sum string by sorting its constituent numbers in ascending order.
 *   This program processes a string representing a sum (e.g., "1+5+3") and outputs the sum
 *   with its numbers sorted (e.g., "1+3+5").
 * * Algorithm: Extraction, Sorting, and Reconstruction.
 *   1. Numbers are extracted from the input string by filtering out '+' characters.
 *   2. The extracted numbers are sorted in ascending order.
 *   3. The sorted numbers are then printed, interleaved with '+' signs.
 * Time Complexity: O(L + M log M) where L is the length of the string and M is the number of integers.
 *   M is at most L/2. The dominant factor is the sorting of numbers.
 * Space Complexity: O(M) for storing the extracted numbers.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to process the arithmetic sum string.
 * Functional Utility: Handles input, extracts and sorts numbers, and prints the sorted sum.
 */
int main() {
  string s1;      // Input string representing the sum.
  int nmbr[105];  // Array to store extracted numbers. Max 105 elements, sufficient for Codeforces constraints.
  int j = 0;      // Index for the 'nmbr' array, also counts the number of extracted numbers.
  cin >> s1;      // Reads the input string.
  int len = s1.length(); // Length of the input string.

  /**
   * Block Logic: Iterates through the input string to extract numerical digits.
   * Invariant: 's1[i]' is the current character being examined. If it's a digit, it's converted and stored.
   */
  for (int i = 0; i < len; i++) {
    /**
     * Condition: Checks if the current character is a digit (not a '+').
     * If true, it extracts the numerical value.
     */
    if (s1[i] != '+') {
      // Inline: Converts a character digit (e.g., '1') to its integer value (e.g., 1).
      nmbr[j] = s1[i] - '0';
      j++; // Increments count of extracted numbers.
    }
  }

  // Algorithm: Sorts the extracted numbers in ascending order.
  // 'nmbr' array is sorted from index 0 up to 'j' (exclusive).
  sort(nmbr, nmbr + j);

  /**
   * Block Logic: Reconstructs and prints the arithmetic sum with sorted numbers.
   * Pre-condition: 'nmbr' array contains 'j' sorted numbers.
   * Invariant: The output string represents the sorted sum with '+' signs between numbers.
   */
  for (int i = 0; i < j; i++) {
    cout << nmbr[i]; // Prints the current sorted number.
    /**
     * Condition: Checks if the current number is not the last one.
     * If true, prints a '+' sign to separate numbers.
     */
    if (i != j - 1) {
      cout << "+";
    }
  }
  cout << endl; // Prints a newline character at the end.
  return 0;     // Indicates successful program execution.
}
