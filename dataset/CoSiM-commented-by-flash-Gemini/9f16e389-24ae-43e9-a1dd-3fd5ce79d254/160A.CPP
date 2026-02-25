/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces (Problem 160A).
 *        The problem asks to find the minimum number of coins to take
 *        from a given set such that the sum of the taken coins is strictly
 *        greater than the sum of the remaining coins.
 * Algorithm: Greedy Approach with Sorting
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 *                  The subsequent greedy selection loop runs in O(N).
 * Space Complexity: O(N) to store the coins in the vector.
 */

#include <bits/stdc++.h> // Includes various standard libraries like iostream, vector, algorithm (for sort).

using namespace std;

/**
 * @brief Main function to solve the Twins problem.
 * @return 0 if the program executes successfully.
 */
int main() {

// Conditional compilation for input/output redirection during local testing.
#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  int n; // Declare an integer to store the number of coins.
  // Read the number of coins.
  cin >> n;
  
  // Declare a vector to store the coin values.
  vector<int> v(n);
  // Initialize 'sum' for the total sum of all coins, and 'sum2' for the sum of selected coins.
  int sum = 0, sum2 = 0;
  
  /**
   * Block Logic: Reads the coin values into the vector and calculates their total sum.
   * Invariant: 'sum' accumulates the value of all coins read so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> v[i];     // Read each coin value.
    sum = sum + v[i]; // Add the coin value to the total sum.
  }
  
  // Algorithm Step: Sort the coins in descending order.
  // `v.rbegin()` and `v.rend()` provide reverse iterators for sorting in descending order.
  sort(v.rbegin(), v.rend());
  
  /**
   * Block Logic: Greedily selects coins starting from the largest until their sum ('sum2')
   *              is strictly greater than half of the total sum ('sum / 2').
   * Invariant: 'sum2' accumulates the value of coins collected, 'sum' represents
   *            the original total sum, and 'i + 1' counts the number of coins collected.
   */
  for (int i = 0; i < n; i++) {
    sum2 = sum2 + v[i]; // Add the current largest coin to the collected sum.
    // Check if the sum of collected coins is strictly greater than the sum of remaining coins.
    // This is equivalent to checking if sum2 > sum - sum2, or 2 * sum2 > sum, or sum2 > sum / 2.
    if (sum2 > sum / 2) {
      cout << i + 1; // Output the minimum number of coins taken.
      return 0;      // Terminate program as the condition is met.
    }
  }

  return 0; // Should not be reached if there is always a solution (which is true for this problem).
}