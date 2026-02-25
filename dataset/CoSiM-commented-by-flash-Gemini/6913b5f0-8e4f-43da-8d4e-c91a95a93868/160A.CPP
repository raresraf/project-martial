/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces (Problem 160A).
 *        The problem asks to find the minimum number of coins to take
 *        from a given set such that the sum of the taken coins is strictly
 *        greater than the sum of the remaining coins.
 * Algorithm: Greedy Approach with Sorting
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 *                  The subsequent greedy selection loop runs in O(N).
 * Space Complexity: O(N) to store the coins.
 */

#include <bits/stdc++.h> // Includes various standard libraries like iostream, algorithm (for sort).

typedef long long ll; // Defines 'll' as a shorthand for long long.

using namespace std;

/**
 * @brief Main function to solve the Twins problem.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Declare variables:
  // n: number of coins.
  // coins[100]: an array to store the coin values (fixed size, potentially unsafe for N > 100).
  // sum: total sum of all coins.
  int n, coins[100], sum = 0;

  // Read the number of coins.
  cin >> n;

  /**
   * Block Logic: Reads the coin values and calculates their total sum.
   * Invariant: 'sum' accumulates the value of all coins read so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> coins[i]; // Read each coin value.
    sum += coins[i];   // Add the coin value to the total sum.
  }

  // Algorithm Step: Sort the coins in ascending order.
  // This allows for a greedy approach by iterating from the largest coins.
  sort(coins, coins + n);
  
  // Initialize 'current_value' to store the sum of coins taken, and 'ans' to count the coins taken.
  int current_value = 0, ans = 0;
  
  /**
   * Block Logic: Greedily selects coins starting from the largest until their sum
   *              exceeds the sum of the remaining coins.
   * Invariant: 'current_value' stores the sum of coins collected, 'sum' stores
   *            the sum of coins remaining, and 'ans' counts the number of coins collected.
   */
  for (int i = n - 1; i >= 0; i--) { // Iterate from the largest coin (end of sorted array).
    // Pre-condition check: if the collected coins already exceed the remaining, print and exit.
    // This optimization can terminate early in some cases, although the problem statement
    // guarantees a solution where current_value will eventually exceed sum.
    if (current_value > sum) {
      cout << ans << endl; // Output the minimum number of coins.
      return 0;            // Terminate program.
    }
    current_value += coins[i]; // Add the current largest coin to the collected sum.
    sum -= coins[i];           // Subtract this coin's value from the total remaining sum.
    ans++;                     // Increment the count of coins taken.
  }
  
  // After the loop, the condition must have been met (or all coins were taken).
  cout << ans << endl; // Output the minimum number of coins.
  return 0;            // Indicate successful program termination.
}

// g++ -std=c++11 -O2 -Wall 160A.cpp -o out