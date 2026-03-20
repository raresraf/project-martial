/**
 * @file 339A.CPP
 * @brief Sorts and reconstructs a sum of numbers (1, 2, or 3) provided as a string.
 * Algorithm: Counting sort. Counts the frequency of each digit (1, 2, 3) and then
 *            reconstructs the sum string by printing them in non-decreasing order.
 * Time Complexity: O(N) where N is the length of the input string, as it involves a single pass
 *                  to count frequencies and a fixed number of operations to print (max 3 distinct digits).
 * Space Complexity: O(1) as the `n` array has a fixed small size.
 */
#include <bits/stdc++.h> // Includes common libraries for competitive programming
using namespace std;

// Array to store the counts of digits 1, 2, and 3.
// n[0] is unused, n[1] stores count of '1', n[2] for '2', n[3] for '3'.
int n[5];

/**
 * @brief Main function to parse the input string, count digit frequencies, and print the sorted sum.
 * Functional Utility: Orchestrates the sorting of digits in a sum string.
 */
int main() {
  // freopen("input.txt", "r", stdin); // Commented out: typically used for redirecting input from a file during development.
  
  string s;
  // Block Logic: Reads the input string representing the sum (e.g., "1+2+3").
  cin >> s;
  
  // Block Logic: Iterates through the input string to count the occurrences of each digit.
  // Pre-condition: Input string contains digits '1', '2', '3' and '+' signs.
  // Invariant: `n[digit]` accurately reflects the frequency of `digit` in the string `s`.
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+') {
      // Converts character digit to integer and increments its count.
      n[s[i] - '0']++;
    }
  }
  
  // Boolean flag to manage printing '+' signs between numbers.
  // Functional Utility: Ensures '+' is printed only between numbers, not before the first or after the last.
  bool f = false;
  
  // Block Logic: Iterates through digits 1, 2, and 3 to print them in sorted order based on their counts.
  // Invariant: Digits are printed in ascending order.
  for (int i = 1; i <= 3; i++) {
    // Prints the current digit 'i' as many times as it appeared in the input string.
    while (n[i]--) {
      if (f) {
        printf("+"); // If not the first number, print a '+' separator.
      }
      f = true;      // Set flag to true after printing the first number.
      printf("%d", i); // Print the digit.
    }
  }
  
  // Functional Utility: Prints a newline character after the sorted sum.
  printf("\n");
  return 0; // Indicates successful execution.
}
