/**
 * @file 110A.CPP
 * @brief This program determines if a given integer 'n' is "nearly lucky".
 * A number is considered nearly lucky if the count of '4's and '7's in its decimal representation
 * is itself either 4 or 7.
 * Algorithm: Digit extraction and counting, followed by a conditional check.
 * Time Complexity: O(log N) where N is the input number, as it iterates through its digits.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h> // Includes various standard library headers, commonly used in competitive programming.
using namespace std;     // Uses the standard namespace to avoid prefixing standard library elements with std::.

int main() {
  long long count, rem, n; // Declares variables: 'count' for lucky digits, 'rem' for remainder, 'n' for input number.
  count = 0;               // Initializes 'count' of lucky digits to zero.
  cin >> n;                // Functional Utility: Reads the input integer 'n'.

  /*
   * Block Logic: Extracts digits from the input number and counts occurrences of '4' or '7'.
   * Invariant: At the start of each iteration, 'n' contains the unprocessed part of the original
   * number, and 'count' accumulates the number of lucky digits found so far. The loop continues
   * until all digits of 'n' have been processed.
   */
  while (n != 0) {
    rem = n % 10; // Extracts the last digit of 'n'.
    n = n / 10;   // Removes the last digit from 'n'.
    // Checks if the extracted digit is a '4' or a '7'.
    if (rem == 4 || rem == 7) {
      count++; // Increments the lucky digit counter if the digit is lucky.
    }
  }

  // Block Logic: Determines if the count of lucky digits is itself a lucky number.
  // Pre-condition: 'count' holds the total number of '4's and '7's in the input 'n'.
  if (count == 4 || count == 7)
    // If 'count' is 4 or 7, the number is "nearly lucky".
    cout << "YES";
  else
    // Otherwise, the number is not "nearly lucky".
    cout << "NO";
  return 0; // Indicates successful program execution.
}