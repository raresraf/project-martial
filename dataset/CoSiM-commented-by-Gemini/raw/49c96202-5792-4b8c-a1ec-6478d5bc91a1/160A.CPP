/**
 * @file 160A.CPP
 * @brief Solution to Codeforces problem 160A (Twins).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * The problem involves splitting a set of coins between two twins. The goal is to
 * find the minimum number of coins one twin must take so that the sum of their
 * coins is strictly greater than the sum of the remaining coins.
 *
 * @section algorithm
 * The problem can be solved using a greedy approach. To minimize the number of
 * coins taken while maximizing their sum, one should always pick the coins with
 * the highest value first.
 *
 * The algorithm is as follows:
 * 1. Read all the coin values into an array.
 * 2. Sort the array of coin values in non-decreasing order.
 * 3. Iterate from k = 1 to n (the number of coins), where k is the number of
 *    coins taken. In each step, assume the k largest coins are taken.
 * 4. Calculate the sum of the k largest coins (the collector's sum) and the
 *    sum of the n-k smallest coins (the other twin's sum).
 * 5. The first value of k for which the collector's sum is strictly greater
 *    than the other twin's sum is the answer.
 *
 * The implementation uses a manual bubble sort (O(n^2)) instead of the more
 * efficient `std::sort` (O(n log n)). It then iteratively calculates the sums.
 *
 * @section complexity
 * - Time Complexity: O(N^2) dominated by the bubble sort implementation. The
 *   subsequent loop to find the answer runs in O(N^2) as well due to the nested
 *   summation loops.
 * - Space Complexity: O(N) to store the coin values in an array.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, temp;
  cin >> n;

  // A C-style array is used. std::vector would be more idiomatic C++.
  int arr[n];
  for (int i = 0; i < n; i++)
    cin >> arr[i];

  /**
   * @brief Sorts the array in ascending order using a bubble sort algorithm.
   *
   * This block arranges the coin values from smallest to largest.
   * Post-condition: arr is sorted.
   */
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (arr[i] > arr[j]) {
        temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
  }

  int total = 0, k = 1, total1 = 0;

  /**
   * @brief Finds the minimum number of coins (k) needed.
   *
   * This loop iterates, increasing the number of coins `k` taken from the
   * largest values. In each step, it compares the sum of the `k` largest
   * coins (`total1`) with the sum of the remaining `n-k` coins (`total`).
   */
  for (int j = 0; j < n; j++) {
    // `total` stores the sum of the n-k smallest coins.
    total = 0;
    for (int i = 0; i < n - k; i++)
      total += arr[i];
    
    // `total1` accumulates the sum of the k largest coins.
    // This inner loop only runs once for i = n-k, adding the next largest coin.
    for (int i = n - k; i > (n - k - 1); i--)
      total1 += arr[i];

    // Check if the collector's sum is strictly greater.
    if (total < total1) {
      cout << k << endl;
      break; // Minimum k found, exit the loop.
    }
    k++; // Increment the number of coins to take.
  }

  return 0;
}
