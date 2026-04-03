/**
 * @8cb882f2-89cb-4f9e-9be9-b0eb57c59386/160A.CPP
 * @brief This program solves the "Twins" problem (Codeforces 160A). Given a set of coins
 * with various values, the objective is to determine the minimum number of coins
 * one person needs to take such that their total sum of coins is strictly greater
 * than the sum of the remaining coins (taken by their "twin"). The optimal strategy
 * is to always take the largest available coins first.
 * Algorithm: Greedy algorithm with sorting.
 * Time Complexity: O(N log N) due to sorting the coins, where N is the number of coins.
 * Space Complexity: O(N) to store the coin values.
 */

#include <algorithm> // Required for std::sort
#include <iostream>  // Required for standard input/output (cin, cout)
#include <numeric>   // Required for std::accumulate
#include <vector>    // Required for std::vector

using namespace std;

int main() {
  // @var n: The total number of coins available.
  int n;
  cin >> n; // Read the number of coins.

  // @var coins: A dynamic array (vector) to store the values of the coins.
  vector<int> coins(n);

  // Block Logic: Read the values of 'n' coins into the vector.
  for (int i = 0; i < n; i++) {
    cin >> coins[i];
  }

  // @var my_sum: Accumulates the sum of coins taken by the current player.
  // @var twin_sum: Accumulates the sum of coins taken by the twin (initially all coins).
  // Block Logic: Initialize sums based on the greedy strategy: twin has all, current player has none.
  int my_sum = 0;
  int twin_sum = accumulate(coins.begin(), coins.end(), 0); // Calculate total sum of all coins.

  // Block Logic: Sort the coins in descending order. This is crucial for the greedy approach.
  // Invariant: The largest coins are processed first.
  sort(coins.rbegin(), coins.rend()); // rbegin() and rend() provide reverse iterators for descending sort.

  // @var num_of_coins: Counts the number of coins taken by the current player.
  int num_of_coins = 0;
  
  // Block Logic: Iterate through the sorted coins, taking the largest ones until the condition is met.
  // Invariant: 'my_sum' increases, 'twin_sum' decreases, 'num_of_coins' increases.
  // Pre-condition: Coins are sorted in descending order.
  for (int i = 0; i < n; i++) {
    // Inline: Take the current largest coin.
    my_sum += coins[i];
    // Inline: Subtract the taken coin's value from the twin's sum.
    twin_sum -= coins[i];
    // Inline: Increment the count of coins taken.
    num_of_coins++;

    // Block Logic: Check if the current player's sum is strictly greater than the twin's sum.
    // If the condition is met, the minimum number of coins has been found.
    if (my_sum > twin_sum) {
      break; // Exit the loop.
    }
  }

  // Post-condition: Print the minimum number of coins required.
  cout << num_of_coins << endl; // Print the result followed by a newline.
  return 0; // Explicitly return 0 for successful execution.
}
