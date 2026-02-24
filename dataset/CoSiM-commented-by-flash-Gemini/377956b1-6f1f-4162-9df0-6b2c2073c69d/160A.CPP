/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A - Twins.
 *
 * This program solves a problem where two twin brothers want to divide a set
 * of coins. One brother wants to take the minimum possible number of coins
 * such that the sum of his coins is strictly greater than the sum of the
 * remaining coins. The objective is to find this minimum number of coins.
 *
 * Algorithm: Greedy approach with sorting. The coins are sorted in descending
 * order, and the first brother greedily takes the largest coins one by one
 * until his sum strictly exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * @brief Main function to execute the Twins problem logic.
   *
   * Reads the number of coins and their values, then sorts the coins.
   * It then greedily selects coins from largest to smallest for one twin
   * until their sum is strictly greater than the sum of the remaining coins.
   * Finally, it prints the count of coins taken by this twin.
   */
  // Declare 'n' for the number of coins.
  int n;
  // Read the total number of coins.
  cin >> n;
  // Declare an array 'cho' to store the coin values.
  int cho[n];
  // Block Logic: Read coin values into the array.
  // Invariant: 'i' iterates from 0 to n-1, populating the 'cho' array.
  for (int i = 0; i < n; i++) {
    cin >> cho[i];
  }

  // Sort the coin values in ascending order.
  // This is a prerequisite for the greedy approach to work by iterating backwards.
  sort(cho, cho + n);
  
  // 'f' will store the sum of coins taken by the first twin.
  int f = 0;
  // 'F' will store the count of coins taken by the first twin.
  int F = 0;
  
  // Block Logic: Greedily select coins for the first twin.
  // Invariant: 'i' iterates from the largest coin (end of sorted array) downwards.
  // Coins are added to 'f' and 'F' increments until 'f' is strictly greater than
  // the sum of remaining coins.
  for (int i = n - 1; i >= 0; i--) {
    f += cho[i]; // Add the current largest coin to the first twin's sum.
    int sum = 0; // Initialize sum for the remaining coins.
    // Block Logic: Calculate the sum of the remaining coins.
    // Invariant: 'j' iterates through coins smaller than the current one being considered for 'f'.
    for (int j = i - 1; j >= 0; j--) {
      sum += cho[j]; // Add remaining coins to 'sum'.
    }
    F++; // Increment the count of coins taken by the first twin.
    // Check if the first twin's sum is strictly greater than the sum of remaining coins.
    if (f > sum) {
      cout << F << endl; // Print the minimum number of coins taken.
      break;             // Exit the loop as the condition is met.
    }
  }

  return 0;
}
