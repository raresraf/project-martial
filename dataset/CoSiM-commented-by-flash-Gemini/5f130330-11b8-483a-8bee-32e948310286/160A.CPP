/**
 * @5f130330-11b8-483a-8bee-32e948310286/160A.CPP
 * @brief This program solves Codeforces problem 160A, "Twins".
 * Given 'n' coin values, the objective is to find the minimum number of coins
 * that can be taken such that their sum is strictly greater than the sum of
 * the remaining coins.
 *
 * Algorithm: Greedy approach. It calculates the total sum, sorts the coins
 * in ascending order, and then iteratively takes coins from the largest
 * end until the sum of taken coins exceeds the sum of remaining coins.
 * Time Complexity: O(N^2) due to the manual bubble sort implementation.
 * Space Complexity: O(1) for storing variables (array size is fixed).
 */

#include <stdio.h>   // For standard C input/output functions like scanf and printf.
#include <string.h>  // Included but not explicitly used in this snippet.
using namespace std; // Uses the standard namespace (implicitly includes iostream sometimes).

int main() {
  // Declares variables: n (number of coins), a (array for coins), i, j (loop counters),
  // S (total sum of coins), k (sum of coins taken), count (number of coins taken).
  int n, a[100], i, j, S = 0, k = 0, count = 1; 
  scanf("%d", &n); // Reads 'n' (number of coins) from standard input.
  
  /**
   * Block Logic: Reads 'n' coin values into array 'a' and calculates their total sum.
   * Invariant: After this loop, 'a' is populated and 'S' holds the total sum of all coins.
   */
  for (i = 0; i < n; i++) {
    scanf("%d", &a[i]); // Reads each coin value.
    S += a[i];           // Adds the coin value to the total sum.
  }
  
  /**
   * Block Logic: Sorts the array 'a' in ascending order using a bubble sort algorithm.
   * This specific swap implementation (`x=x+y; y=x-y; x=x-y;`) is a common trick
   * to swap without a temporary variable, but is not generally recommended due to
   * potential overflow and lack of clarity.
   * Invariant: After this nested loop, 'a' is sorted in non-decreasing order.
   * Time Complexity: O(N^2).
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < n - 1; j++) {
      if (a[j] > a[j + 1]) { // If current element is greater than the next element.
        a[j] = a[j] + a[j + 1];     // Swap without temporary variable.
        a[j + 1] = a[j] - a[j + 1]; // This is `a[j+1] = (a[j]+a[j+1]) - a[j+1] = a[j]` (original a[j]).
        a[j] = a[j] - a[j + 1];     // This is `a[j] = (a[j]+a[j+1]) - a[j]` (original a[j+1]).
      }
    }
  }

  /**
   * Block Logic: Iterates through the sorted coins from largest to smallest, accumulating
   * their sum ('k') and counting how many coins are needed ('count') until 'k' is
   * strictly greater than the sum of the remaining coins ('S - k').
   * Invariant: 'count' stores the minimum number of coins needed to satisfy the condition.
   */
  for (i = 0; i < n; i++) {
    // Inline: Adds the i-th largest coin to 'k'. 'a[n - i - 1]' accesses elements from the end of the ascending sorted array.
    k += a[n - i - 1]; 
    // Inline: Checks if the sum of taken coins ('k') is still not strictly greater than the sum of remaining coins ('S - k').
    if (k <= S - k)
      count++; // Increments 'count' if the condition is not yet met.
    else
      break; // Breaks loop if 'k' is strictly greater than 'S - k'.
  }
  printf("%d", count); // Prints the minimum number of coins taken.
  return 0; // Indicates successful program termination.
}