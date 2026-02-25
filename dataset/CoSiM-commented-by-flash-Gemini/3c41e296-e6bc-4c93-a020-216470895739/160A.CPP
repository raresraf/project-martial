/**
 * @file 160A.CPP
 * @brief This program solves the "Twins" problem, where the goal is to find the minimum number of coins
 * whose sum is strictly greater than the sum of the remaining coins.
 * It's a classic greedy algorithm problem.
 * Algorithm: Greedy approach after sorting. Sort all coins in descending order and iteratively
 * pick the largest coins until their sum exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) due to the sorting step, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, including algorithm for sorting.
using namespace std;
int main() {
  // Functional Utility: Optimizes C++ standard I/O streams for competitive programming.
  // Disables synchronization with C standard I/O and unties cin/cout for faster operations.
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int n; // Declares an integer 'n' to store the number of coins.
  cin >> n; // Reads the number of coins.
  int arr[n]; // Declares a dynamic array 'arr' to store the values of the coins.
  int sum = 0; // Initializes 'sum' to store the total sum of all coins.
  int money = 0; // Initializes 'money' to store the sum of coins collected by the current player.
  int countt = 0; // Initializes 'countt' to store the number of coins collected by the current player.
  /**
   * Block Logic: Reads the values of 'n' coins into the array 'arr' and calculates their total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i]; // Reads each coin's value.
    sum = sum + arr[i]; // Accumulates the total sum.
  }
  /**
   * Functional Utility: Sorts the array 'arr' in ascending order.
   */
  sort(arr, arr + n);
  /**
   * Functional Utility: Reverses the sorted array 'arr', effectively sorting it in descending order.
   * This is crucial for the greedy approach, ensuring the largest coins are considered first.
   */
  reverse(arr, arr + n);
  /**
   * Block Logic: Iterates through the coins (from largest to smallest) to find the minimum number
   * whose sum ('money') is strictly greater than the sum of the remaining coins ('sum').
   * Invariant: 'money' increases, 'sum' decreases, and 'countt' tracks the number of coins taken.
   */
  for (int i = 0; i < n; i++) {
    money = money + arr[i]; // Adds the current largest coin to the player's collection.
    sum = sum - arr[i]; // Subtracts the current largest coin from the total sum of remaining coins.
    countt++; // Increments the count of coins taken.
    // Conditional Logic: Checks if the collected money is strictly greater than the sum of remaining coins.
    if (money > sum) {
      break; // If the condition is met, no more coins need to be taken.
    }
  }
  cout << countt; // Prints the minimum number of coins required.
  return 0; // Indicates successful program execution.
}
