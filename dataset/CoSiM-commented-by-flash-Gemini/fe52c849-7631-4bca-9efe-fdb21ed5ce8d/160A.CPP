/**
 * @file 160A.CPP
 * @brief This competitive programming solution addresses the "Twins" problem,
 *        where the goal is to find the minimum number of coins one needs to take
 *        from a given set such that the sum of the taken coins is strictly
 *        greater than the sum of the remaining coins.
 * Algorithm: Greedy approach combined with sorting.
 *            1. Read all coin values and calculate their total sum.
 *            2. Sort the coins in descending order. This is crucial for the greedy strategy
 *               as taking the largest coins first ensures the minimum count.
 *            3. Iterate through the sorted coins, accumulating their sum.
 *            4. At each step, check if the accumulated sum of taken coins is
 *               strictly greater than the sum of the coins not yet taken.
 *            5. The moment this condition is met, the current count of taken coins is the answer.
 * Time Complexity: O(N log N) due to the sorting step, where N is the number of coins.
 *                  The subsequent linear scan is O(N).
 * Space Complexity: O(N) for storing the coin values in a vector.
 */
/*
    Problem Link    :   https://codeforces.com/problemset/problem/160/A
    Contest Link    :   https://codeforces.com/contest/160/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h> // Includes common standard library headers, including vector, algorithm, iostream.
using namespace std;

/**
 * @brief Main function of the program. Solves the "Twins" problem.
 * Functional Utility: Serves as the entry point, orchestrating the greedy coin selection process.
 */
int main() {
  int n; // Declares an integer to store the number of coins.
  cin >> n; // Reads the number of coins.
  vector<int> a(n); // Declares a vector to store the coin values.
  int sum = 0;      // Stores the total sum of all coin values.

  // Block Logic: Reads coin values into the vector and calculates their total sum.
  // Invariant: After this loop, 'a' contains all coin values, and 'sum' holds their total.
  for (int i = 0; i < n; i++) {
    cin >> a[i]; // Reads each coin value.
    sum += a[i]; // Accumulates the total sum.
  }

  sort(a.begin(), a.end());      // Sorts the vector 'a' in ascending order.
  reverse(a.begin(), a.end()); // Reverses the sorted vector, placing coins in descending order.
                               // Functional Utility: This greedy strategy prioritizes taking larger coins.
  int mx = 0; // Stores the accumulated sum of coins taken.
  int i = 0;  // Loop counter, also represents the number of coins taken.

  // Block Logic: Iterates through the coins from largest to smallest, accumulating their sum
  //              until the sum of taken coins exceeds the sum of remaining coins.
  // Invariant: At each step, `mx` is the sum of the `i` largest coins, and the loop continues
  //            as long as `mx` is not strictly greater than `sum - mx`.
  for (i = 0; i < n; i++) {
    mx += a[i]; // Add the current largest coin to the sum of taken coins.
    // Block Logic: Checks if the sum of taken coins (`mx`) is strictly greater than
    //              the sum of the remaining coins (`sum - mx`).
    if (mx > sum - mx)
      break; // If the condition is met, break the loop.
  }
  cout << i + 1 << endl; // Prints the count of coins taken (i + 1 because i is 0-indexed).
  return 0;              // Indicates successful program termination.
}
