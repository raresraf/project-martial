/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A - Twins.
 *
 * This program solves a problem where two twin brothers want to divide a set
 * of coins. One brother wants to take the minimum possible number of coins
 * such that the sum of his coins is strictly greater than the sum of the
 * remaining coins. The objective is to find this minimum number of coins.
 *
 * Algorithm: Greedy approach with sorting. The coins are sorted in ascending
 * order. The first brother then iterates from the largest coin (end of sorted
 * array) and greedily adds coins to his sum until his sum is strictly greater
 * than the sum of the remaining coins.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values.
 */

#include <algorithm> // Required for std::sort.
#include <iostream>  // Required for standard input/output (cin, cout).
#include <stdio.h>   // Included but not explicitly used in this solution.
using namespace std; // Uses the standard namespace.

main() { // Using 'main' without return type defaults to 'int main()' in C++ standard.
  /**
   * @brief Main function to execute the Twins problem logic.
   *
   * Reads the number of coins 'n' and then their individual values.
   * It calculates the total sum of all coins and sorts the coins in ascending order.
   * It then greedily selects coins starting from the largest for one twin until
   * their sum is strictly greater than the sum of the remaining coins.
   * Finally, it determines and prints the count of coins taken by this twin.
   *
   * Note: The variable `ans1` is implicitly initialized to an undefined value,
   * which can lead to unexpected behavior. For correctness, it should be
   * explicitly initialized to 0.
   */
  // Declare integer variables:
  // 'n' for the number of coins.
  // 'ans1' for the sum of coins taken by the first twin (uninitialized).
  // 'ans2' for the sum of coins remaining for the second twin.
  // 'ans' to store the index or count of coins.
  // 'i' for loop counter.
  int n, ans1, ans2, ans, i;
  // Read the total number of coins.
  cin >> n;
  // Declare a dynamic array 'a' of size 'n' to store coin values.
  int a[n];
  // Initialize 'sum' to calculate the total sum of all coins.
  int sum = 0;
  // Block Logic: Read coin values and calculate their total sum.
  // Invariant: 'i' iterates from 0 to n-1, reading each coin and adding it to 'sum'.
  for (i = 0; i < n; i++) {
    cin >> a[i];   // Read current coin value.
    sum += a[i];   // Add to total sum.
  }

  // Sort the coin values in ascending order.
  // This is a prerequisite for the greedy approach which will iterate backwards.
  sort(a, a + n);

  // Block Logic: Greedily select coins for the first twin.
  // Invariant: 'i' iterates from the largest coin (end of sorted array) downwards.
  // 'ans1' accumulates the current twin's coins, 'ans2' is the sum of remaining coins.
  // The loop continues until 'ans1' is strictly greater than 'ans2'.
  for (i = n - 1; i >= 0; i--) {
    ans1 += a[i];       // Add the current largest coin to the first twin's sum.
    ans2 = sum - ans1;  // Calculate the sum of coins remaining for the second twin.

    // If the first twin's sum is strictly greater than the sum of remaining coins.
    if (ans1 > ans2) {
      // The number of coins taken is (n - i) since 'i' is 0-indexed from the start.
      // So, 'ans' is effectively 'n - (current_index_from_start_of_iteration)'
      // which corresponds to 'n - i' if 'i' started from 'n-1'.
      // More precisely, if 'i' is the *last* index added, then 'n - i' coins were added.
      ans = n - i; // Calculate the count of coins taken.
      break;       // Exit the loop as the condition is met.
    }
  }
  // Output the minimum number of coins ('ans') that satisfy the condition.
  cout << ans << endl;
  return 0; // Indicate successful program execution.
}
