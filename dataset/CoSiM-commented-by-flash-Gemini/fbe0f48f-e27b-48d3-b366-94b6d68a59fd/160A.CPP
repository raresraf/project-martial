/**
 * @file 160A.CPP
 * @brief Solves a problem related to "lucky numbers" where numbers are considered
 *        lucky if they contain a certain count of lucky digits (4s and 7s).
 *        The program reads 'N' integers, and for each integer, it counts the
 *        occurrences of digits '4' and '7'. If this count is less than or
 *        equal to a given threshold 'K', the integer is considered "lucky"
 *        and added to a total count.
 * Algorithm: Digit Extraction, Counting, and Conditional Summation
 * Time Complexity: O(N * log10(Max_M)), where N is the number of integers processed,
 *                  and log10(Max_M) is the number of digits in the maximum integer.
 * Space Complexity: O(1).
 */

#include <algorithm> // Required for std::sort, etc. (though not directly used in this snippet).
#include <bitset>    // For bitset operations.
#include <cmath>     // For mathematical functions.
#include <cstdio>    // For standard C input/output functions like scanf, printf.
#include <cstdlib>   // For general utilities.
#include <cstring>   // For C-style string functions.
#include <ctime>     // For time-related functions.
#include <deque>     // For deque container.
#include <functional> // For function objects.
#include <iomanip>    // For output formatting.
#include <iostream>   // For standard C++ input/output (cin, cout).
#include <numeric>    // For numeric operations.
#include <queue>      // For queue container.
#include <set>        // For set container.
#include <sstream>    // For string stream manipulation.
#include <stack>      // For stack container.
#include <utility>    // For std::pair.

using namespace std;

/**
 * @brief Main function to process numbers, count lucky digits, and sum up "lucky" numbers.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Declare variables:
  // n: total number of integers to process.
  // k: the threshold for the count of lucky digits.
  // c: counter for lucky digits in the current number being processed.
  // s: total count of numbers satisfying the lucky digit condition.
  // i: loop counter.
  // m: the current integer being processed.
  // d: extracted digit from 'm'.
  int n, k, c, s, i, m, d;
  
  // Functional Utility: Redirects standard input from "in.txt" file.
  // This is a common practice in competitive programming for local testing.
  freopen("in.txt", "r", stdin);
  
  /**
   * Block Logic: Continuously reads pairs of 'n' and 'k' and processes sets of numbers.
   * Invariant: The loop continues as long as valid input pairs for 'n' and 'k' are provided.
   */
  while (scanf("%d %d", &n, &k) == 2) { // Read N and K. `scanf` returns the number of items successfully read.
    s = 0; // Reset the total count of lucky numbers for each test case.
    
    /**
     * Block Logic: Iterates 'n' times to process each integer.
     * Invariant: 's' accumulates the count of numbers whose lucky digit count <= k.
     */
    for (i = 0; i < n; i++) {
      scanf("%d ", &m); // Read the current integer.
      c = 0; // Reset the lucky digit counter for the current integer.
      
      /**
       * Block Logic: Extracts digits from 'm' and counts occurrences of '4' or '7'.
       * Invariant: 'm' is progressively reduced by removing its last digit, 'c' accumulates lucky digit count.
       */
      while (m != 0) { // Loop until all digits of 'm' have been processed.
        d = m % 10; // Inline: Get the last digit of 'm'.
        m /= 10;    // Inline: Remove the last digit from 'm'.
        // Check if the extracted digit 'd' is a lucky digit (4 or 7).
        if (d == 7 || d == 4) {
          c++; // If it is, increment the lucky digit counter.
        }
      }
      // If the count of lucky digits 'c' is less than or equal to the threshold 'k'.
      if (c <= k)
        s++; // Increment the total count of numbers satisfying the condition.
    }
    printf("%d\n", s); // Print the total count 's' followed by a newline.
  }
  return 0; // Indicate successful program termination.
}
