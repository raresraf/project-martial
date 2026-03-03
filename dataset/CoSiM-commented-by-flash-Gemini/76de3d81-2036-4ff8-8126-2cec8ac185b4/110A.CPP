/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A - "Lucky Numbers".
 * This program determines if a given number 'n' is "nearly lucky".
 * A number is nearly lucky if the count of "lucky digits" (4 and 7) in its
 * decimal representation is itself a lucky number (i.e., contains only 4s and 7s).
 * Algorithm:
 * 1. Count the occurrences of digits 4 and 7 in the input number 'n'.
 * 2. Then, count the occurrences of digits 4 and 7 in the resulting count.
 * 3. Compare this second count with the total digits in the first count to check if it's a "lucky number".
 * Time Complexity: O(log N) - where N is the input number. The number of operations is proportional
 *                  to the number of digits in N, and then to the number of digits in the count of lucky digits.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
#include <iostream> // Standard input/output stream library
using namespace std; // Use the standard namespace to avoid prefixing std::

int main() {
  // n: stores the input number.
  // i: a copy of n used for digit extraction in the first loop.
  // c: counts the lucky digits (4 or 7) in the input number 'n'.
  // r: stores the current digit being processed.
  // m: a flag, set to 0 if 'n' contains no lucky digits initially.
  // k: counts the total digits in 'c'.
  // l: counts the lucky digits (4 or 7) in 'c'.
  long long int n, i, c = 0, r = 0, m = 1;
  
  // Reads the input number 'n'.
  cin >> n;
  i = n; // Initialize 'i' with the value of 'n' for digit extraction.
  
  /**
   * Block Logic: Extracts digits from the input number 'n' and counts lucky digits.
   * This loop iterates through each digit of 'n' from right to left.
   * Invariant: 'c' accumulates the count of lucky digits found so far.
   */
  while (i != 0) {
    r = i % 10; // Get the last digit of 'i'.
    // Checks if the extracted digit is either 4 or 7.
    if (r == 4 || r == 7) {
      c++; // If it's a lucky digit, increment the lucky digit count.
    }
    i = i / 10; // Remove the last digit from 'i'.
  }
  
  r = 0; // Reset 'r' for subsequent use.
  
  // If no lucky digits were found in 'n' (c is 0), set flag 'm' to 0.
  // This helps distinguish between a count of 0 (no lucky digits) and a lucky count.
  if (c == 0) {
    m = 0;
  }
  
  // k: will count the total number of digits in 'c'.
  // l: will count the number of lucky digits (4 or 7) within 'c'.
  long long int k = 0, l = 0;
  
  /**
   * Block Logic: Extracts digits from the count 'c' and determines if 'c' is a lucky number.
   * This loop iterates through each digit of the count 'c'.
   * Invariant: 'l' accumulates lucky digits within 'c', and 'k' counts total digits in 'c'.
   */
  while (c != 0) {
    r = c % 10; // Get the last digit of 'c'.
    k++;        // Increment total digit count of 'c'.
    // Checks if the extracted digit from 'c' is either 4 or 7.
    if (r == 4 || r == 7) {
      l++; // If it's a lucky digit, increment the lucky digit count for 'c'.
    }
    c /= 10; // Remove the last digit from 'c'.
  }
  
  // Final Decision Logic: Determines if the original number 'n' is "nearly lucky".
  // A number is "nearly lucky" if:
  // 1. It contained at least one lucky digit (m != 0, so c was not 0).
  // 2. The count of its lucky digits (c) is itself a lucky number (l == k, meaning all digits of c were lucky).
  if (l != k || m == 0) {
    // If 'c' is not a lucky number (l != k) or 'n' had no lucky digits (m == 0).
    cout << "NO"; // Output "NO".
  } else {
    // If 'c' is a lucky number and 'n' had lucky digits.
    cout << "YES"; // Output "YES".
  }
}

