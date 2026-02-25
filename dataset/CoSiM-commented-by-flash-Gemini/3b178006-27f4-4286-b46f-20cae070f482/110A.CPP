/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from Codeforces (Problem 110A).
 *        This program reads a large integer, counts the occurrences of "lucky digits"
 *        (4 and 7) within it, and then determines if this count itself is a lucky number.
 *        This version processes the number using arithmetic operations.
 * Algorithm: Digit Extraction and Counting
 * Time Complexity: O(log10 N), where N is the input number (proportional to the number of digits).
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h> // Includes various standard libraries.
using namespace std;

typedef long long int ll; // Defines 'll' as a shorthand for long long int to handle large numbers.

/**
 * @brief Main function to check if an input number is a "nearly lucky number".
 * @return 0 if the program executes successfully.
 */
int main() {
  ll n; // Declare a long long integer to store the input number.
  // Read the input number from standard input.
  cin >> n;
  
  // Initialize 'k' with 'n' for digit extraction and 'c' as the lucky digit counter.
  ll k = n, r, c = 0; 
  
  /**
   * Block Logic: Extracts digits from the number 'k' and counts occurrences of '4' or '7'.
   * Invariant: 'k' is progressively reduced by removing its last digit, 'c' accumulates lucky digit count.
   */
  while (k > 0) { // Loop until all digits of 'k' have been processed.
    r = k % 10; // Inline: Get the last digit of 'k'.
    // Check if the extracted digit 'r' is a lucky digit (4 or 7).
    if (r == 4 || r == 7)
      c++; // If it is, increment the lucky digit counter.
    k /= 10; // Inline: Remove the last digit from 'k'.
  }
  
  /**
   * Block Logic: Determines if the count of lucky digits ('c') is itself a lucky number.
   * Pre-condition: 'c' holds the total count of lucky digits in the input number.
   * Post-condition: Prints "YES" if 'c' is 4 or 7, otherwise prints "NO".
   */
  if (c == 4 || c == 7) // Check if the count 'c' is 4 or 7.
    cout << "YES\n";    // If so, the number is "nearly lucky".
  else
    cout << "NO\n";     // Otherwise, it is not.
  
  return 0; // Indicate successful program termination.
}