/**
 * @file 160A.CPP
 * @brief Solution to Codeforces Problem 160A: "Twins".
 *
 * This problem involves two twin brothers who want to divide a collection of coins.
 * They want to take coins such that each twin's sum of coins is strictly greater
 * than the sum of the remaining coins, and the first twin wants to take the minimum
 * number of coins possible to achieve this.
 *
 * Algorithm: Greedy approach with sorting.
 * The algorithm first calculates the total sum of all coins. Then, it sorts the coins
 * in descending order. The first twin greedily takes the largest coins one by one,
 * summing them up, until their sum is strictly greater than half of the total sum.
 * The number of coins taken is the answer.
 *
 * Time Complexity: O(N log N) due to the sorting step, where N is the number of coins.
 *                  The subsequent loop runs at most N times.
 * Space Complexity: O(N) for storing the coins in an array (or O(1) if modifying input array).
 */
#include <algorithm> // Required for std::sort.
#include <iostream>  // Required for input/output operations (cin, cout).
using namespace std; // Uses the standard namespace to avoid prefixing std::

/**
 * @brief Main function to solve the "Twins" problem.
 * Reads the number of coins and their values, then calculates and prints
 * the minimum number of coins the first twin needs to take.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
  int n; // Stores the number of coins.
  int num = 0; // Counts the number of coins the first twin takes.
  int coin[101]; // Array to store the coin values. Maximum 100 coins.
  int total = 0; // Stores the sum of all coins.
  int sum = 0; // Stores the sum of coins taken by the first twin.
  
  cin >> n; // Reads the number of coins.
  
  /**
   * Block Logic: Reads coin values from input and calculates the total sum of all coins.
   * Pre-condition: 'n' is the number of coins to read.
   * Invariant: 'total' accumulates the sum of all coin values.
   */
  for (int i = 0; i < n; ++i) {
    cin >> coin[i]; // Reads each coin value.
    total += coin[i]; // Adds the coin value to the total sum.
  }
  
  // Block Logic: Sorts the coins in ascending order.
  // This is a pre-requisite for the greedy strategy, as we will iterate in reverse
  // to pick the largest coins first.
  sort(coin, coin + n);
  
  /**
   * Block Logic: Greedily takes coins from largest to smallest until the first twin's sum
   * is strictly greater than half of the total sum.
   * Pre-condition: 'coin' array is sorted, 'sum' and 'num' are zero.
   * Invariant: 'sum' accumulates the coins taken by the first twin, 'num' counts them.
   */
  for (int i = n - 1; i >= 0; i--) {
    sum += coin[i]; // Adds the largest available coin to the first twin's sum.
    num += 1; // Increments the count of coins taken.
    // Block Logic: Checks if the first twin's sum is strictly greater than half the total.
    if (sum * 2 > total) { // Equivalent to sum > total / 2
      cout << num; // Prints the minimum number of coins taken.
      return 0; // Exits the program as the condition is met.
    }
  }
}
