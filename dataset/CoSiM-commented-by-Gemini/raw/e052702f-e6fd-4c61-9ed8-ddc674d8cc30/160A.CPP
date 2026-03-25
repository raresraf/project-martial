#include <algorithm>
#include <iostream>
#include <vector> // For std::vector
#include <numeric> // For std::accumulate
using namespace std;

/**
 * @file 160A.CPP
 * @brief Solution for a greedy algorithm problem, "Twins".
 *
 * Problem Statement:
 * Given a set of coins of varying values, find the minimum number of coins
 * you need to take such that the sum of the values of your coins is strictly
 * greater than the sum of the values of the remaining coins.
 *
 * Algorithm:
 * The optimal strategy is a greedy one: to achieve the target sum with the
 * fewest coins, you should always pick the coins with the largest values first.
 *
 * 1.  Read all coin values and calculate their total sum.
 * 2.  Sort the coins in descending (largest to smallest) order.
 * 3.  Iterate through the sorted coins, adding them one by one to a temporary
 *     sum (`tem`) and counting the number of coins taken (`i`).
 * 4.  Stop as soon as the accumulated sum `tem` is strictly greater than half
 *     of the total sum.
 * 5.  The number of coins taken at that point is the minimum required.
 *
 * @note This code uses a Variable Length Array (VLA) `int coin[n]`, which is
 * a non-standard C++ extension. A `std::vector` is the standard-compliant
 * alternative.
 *
 * Time Complexity: O(N log N), dominated by the sort operation.
 * Space Complexity: O(N) to store the coin values.
 */
int main() {
  int n;
  int i = 0;
  int sum = 0; // The total sum of all coins.
  int tem = 0; // The sum of coins taken by the first twin.
  cin >> n;

  // A VLA is a non-standard C++ extension.
  int coin[n];

  // Block Logic: Read coin values and calculate the total sum.
  while (i < n) {
    cin >> coin[i];
    sum += coin[i];
    i++;
  }

  // Sort the coins in descending order.
  sort(coin, coin + n, greater<int>());

  // Reset counter 'i' to be used for counting coins taken.
  i = 0;
  // Block Logic: Greedily take the largest coins until the sum is sufficient.
  // Invariant: 'tem' is the sum of the 'i' largest coins.
  // The loop continues as long as our sum is not strictly greater than the other half.
  while (tem <= sum / 2) {
    // Take the next largest coin.
    tem += coin[i];
    // Increment the count of coins taken.
    i++;
  }

  // 'i' now holds the minimum number of coins needed.
  cout << i;
  return 0;
}