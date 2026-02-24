/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A - Nearly Lucky Number.
 *
 * This program determines if a given positive integer 'n' is a "nearly lucky" number.
 * A number is considered "lucky" if it contains only the digits 4 and 7.
 * A number is "nearly lucky" if the count of lucky digits (4s or 7s) within it
 * is itself a lucky number (i.e., the count is 4 or 7).
 *
 * Algorithm: Digit extraction and counting. The program iteratively extracts
 * digits from the input number and counts how many of them are 4 or 7. Then,
 * it checks if this count is either 4 or 7.
 * Time Complexity: O(log10(N)), where N is the input number, as it iterates
 *                  through each digit.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h> // Includes various standard libraries (e.g., iostream, string, vector, algorithm).

using namespace std; // Uses the standard namespace.

typedef long long ll;   // Alias for long long integer.
typedef vector<int> vi; // Alias for vector of integers.

/**
 * @brief Checks if a given number is "nearly lucky".
 *
 * A number is nearly lucky if the count of lucky digits (4 or 7) within it
 * is itself a lucky number (4 or 7).
 *
 * @param n The input number to check.
 * @return "YES" if the number is nearly lucky, "NO" otherwise.
 */
string solve(ll n) {
  // Initialize a counter for lucky digits (4s and 7s).
  int cnt = 0;
  // Block Logic: Extract digits and count lucky ones.
  // Invariant: 'n' is progressively reduced by removing its last digit.
  // The loop continues as long as 'n' has digits remaining.
  while (n > 0) {
    // Check if the last digit of 'n' is 4 or 7.
    if (n % 10 == 4 || n % 10 == 7) {
      cnt++; // Increment lucky digit count.
    }
    n /= 10; // Remove the last digit from 'n'.
  }

  // Block Logic: Determine if the count of lucky digits is itself a lucky number.
  // Pre-condition: 'cnt' holds the total count of lucky digits in the original number.
  return (cnt == 4 || cnt == 7) ? "YES" : "NO";
}

int main() {
  /**
   * @brief Main function to execute the Nearly Lucky Number problem logic.
   *
   * Optimizes C++ standard stream I/O for faster execution.
   * Reads a long long integer 'n', calls the `solve` function to determine
   * if it's a nearly lucky number, and prints the result.
   */
  // Functional Utility: Decouples C++ streams from standard C streams and flushes
  // to optimize high-throughput I/O often required for competitive programming constraints.
  ios::sync_with_stdio(false);
  cin.tie(NULL); // Unties cin from cout.

  // Declare a long long integer to store the input number.
  ll n;
  // Read the input number.
  cin >> n;
  // Call the solve function and print its result, followed by a newline.
  cout << solve(n) << endl;
  return 0; // Indicate successful program execution.
}
