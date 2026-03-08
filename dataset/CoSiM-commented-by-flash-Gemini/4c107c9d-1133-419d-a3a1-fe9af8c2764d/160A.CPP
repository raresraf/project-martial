/**
 * @file 160A.CPP
 * @brief Determines the minimum number of largest coins required such that their sum is strictly greater than the sum of the remaining coins.
 *
 * Algorithm: Greedy approach. Sorts coins in descending order and iteratively picks the largest ones.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

// Array to store coin values. Maximum 101 coins.
int arr[101];

/**
 * @brief Main function to solve the twin coins problem.
 * @pre Input consists of an integer 'n' (number of coins), followed by 'n' integers representing coin values.
 * @post Prints the minimum number of coins whose sum is strictly greater than the sum of the rest.
 */
int main() {
  int n, sum = 0, sum1 = 0, sum2 = 0, flag = 0; // 'n': number of coins, 'sum': total sum of all coins, 'sum1': sum of chosen coins, 'sum2': sum of remaining coins, 'flag': indicates if condition is met.

  // Functional Utility: Reads the number of coins.
  cin >> n;

  /**
   * Block Logic: Reads coin values into an array and calculates their total sum.
   * Invariant: After each iteration, 'arr[i]' holds a coin value, and 'sum' accumulates the total.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }

  // Functional Utility: Sorts the coins in ascending order to facilitate picking the largest ones first.
  sort(arr, arr + n);

  /**
   * Block Logic: Iterates from the largest coins, adding them to 'sum1' until 'sum1' is greater than 'sum2'.
   * Invariant: 'sum1' increases with each iteration by adding the next largest coin. 'sum2' decreases accordingly.
   *            The loop terminates when 'sum1' becomes strictly greater than 'sum2'.
   */
  for (int i = 0; i < n; i++) {
    // Calculates the sum of remaining coins after taking 'i' largest coins.
    sum2 = sum - arr[n - i - 1] - sum1;
    // Adds the current largest coin to 'sum1'.
    sum1 += arr[n - 1 - i];

    /**
     * Block Logic: Checks if the sum of chosen coins ('sum1') is greater than the sum of remaining coins ('sum2').
     * Pre-condition: 'sum1' and 'sum2' are updated for the current set of chosen coins.
     * Invariant: If the condition is met, the number of chosen coins ('i + 1') is printed, and the loop terminates.
     */
    if (sum1 > sum2) {
      cout << i + 1; // Prints the count of coins.
      flag = 1; // Sets flag to indicate condition met.
      break;    // Terminates the loop.
    }
  }

  /**
   * Block Logic: Handles the edge case where the condition might not be met within the loop (e.g., only one coin, or all coins needed).
   * Pre-condition: 'flag' indicates if the condition was met and printed inside the loop.
   * Invariant: If 'flag' is 0, it means all coins were needed to satisfy the condition (or the loop finished without 'sum1 > sum2' for some reason, though in this problem, it should always be met by 'n' coins at most).
   */
  if (flag == 0)
    cout << n; // Prints 'n' if all coins are needed (should ideally not be reached with proper logic but for safety).
  // Indicates successful program execution.
  return 0;
}
