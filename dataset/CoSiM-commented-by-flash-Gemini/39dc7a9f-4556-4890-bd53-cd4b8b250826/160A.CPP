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
 * array) and greedily adds coins to his sum (`s2`) while subtracting them
 * from the total (`s1`, representing the other twin's sum). The process stops
 * when `s2` strictly exceeds `s1`.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values.
 */

#include <bits/stdc++.h> // Includes various standard libraries (e.g., iostream, vector, algorithm).
using namespace std;     // Uses the standard namespace.

// Declare global integer variables to store the total sum of all coins (s1)
// and the sum of coins taken by the first twin (s2).
int s1, s2;

int main() {
  /**
   * @brief Main function to execute the Twins problem logic.
   *
   * Reads the number of coins 'n' and then their individual values.
   * It first calculates the total sum of all coins. Then, it sorts the coins
   * in ascending order. It then greedily picks coins starting from the largest
   * for the first twin, updating both twins' sums, until the first twin's sum
   * is strictly greater than the second twin's sum. Finally, it prints the
   * minimum number of coins taken by the first twin.
   */
  // Declare 'n' for the number of coins.
  int n;
  // Read the total number of coins.
  cin >> n;
  // Declare an array 'a' to store the coin values (max 1000 coins).
  int a[1000];
  // Block Logic: Read coin values and calculate their initial total sum.
  // Invariant: 'i' iterates from 0 to n-1, reading each coin and adding it to 's1'.
  for (int i = 0; i < n; i++) {
    cin >> a[i];     // Read current coin value.
    s1 += a[i];      // Add to the total sum of all coins.
  }

  // Sort the coin values in ascending order.
  // This is a prerequisite for the greedy approach which will iterate backwards.
  sort(a, a + n);
  
  // Block Logic: Greedily select coins for the first twin.
  // Invariant: 'i' iterates from the largest coin (end of sorted array) downwards.
  // 's2' accumulates the current twin's coins, and 's1' represents the remaining coins.
  // The loop continues until 's2' is strictly greater than 's1'.
  for (int i = n - 1; i >= 0; i--) {
    s2 += a[i]; // Add the current largest coin to the first twin's sum.
    s1 -= a[i]; // Subtract this coin from the remaining total sum.
    // If the first twin's sum (s2) is strictly greater than the second twin's sum (s1).
    if (s2 > s1) {
      // The number of coins taken is (n - i), as 'i' is the 0-based index
      // of the last coin taken from the sorted array.
      cout << n - i;
      break; // Exit the loop as the condition is met.
    }
  }
  return 0; // Indicate successful program execution.
}