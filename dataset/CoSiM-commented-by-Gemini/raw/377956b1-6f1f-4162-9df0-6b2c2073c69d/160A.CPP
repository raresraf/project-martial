/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem with a O(N^2) greedy approach.
 *
 * The problem is to find the minimum number of coins to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins.
 *
 * Algorithm:
 * 1. Read all coin values.
 * 2. Sort the coins in ascending order.
 * 3. Greedily take the largest coins one by one (by iterating from the end of
 *    the sorted array).
 * 4. In each step, add the current coin to your sum and recalculate the sum of
 *    all remaining coins.
 * 5. Compare your sum with the remaining sum. If it's strictly greater, the
 *    loop terminates, and the number of coins taken is the answer.
 *
 * @note This implementation is inefficient. The sum of the remaining coins is
 * recalculated in every iteration of the main loop, leading to a time complexity
 * of O(N^2). A more optimal solution would pre-calculate the total sum and have
 * a single loop, achieving O(N log N) complexity dominated by the sort.
 *
 * Time Complexity: O(N^2), due to the nested loop for sum calculation.
 * Space Complexity: O(N) to store the coin values.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int cho[n]; // Array to store coin values.
  for (int i = 0; i < n; i++) {
    cin >> cho[i];
  }

  // Sort the coins in ascending order.
  sort(cho, cho + n);
  
  int f = 0; // Sum of coins taken by the current twin.
  int F = 0; // Counter for the number of coins taken.

  /**
   * Block Logic: Greedily take the largest coins.
   * This loop iterates from the largest coin downwards.
   */
  for (int i = n - 1; i >= 0; i--) {
    f += cho[i]; // Add the largest available coin to our hand.
    
    // Inefficiently recalculate the sum of all remaining coins.
    int sum = 0;
    for (int j = i - 1; j >= 0; j--) {
      sum += cho[j];
    }
    
    F++; // Increment the count of coins taken.

    // Check if our sum is now strictly greater than the other twin's sum.
    if (f > sum) {
      cout << F << endl;
      break; // Minimum number of coins found.
    }
  }

  return 0;
}
