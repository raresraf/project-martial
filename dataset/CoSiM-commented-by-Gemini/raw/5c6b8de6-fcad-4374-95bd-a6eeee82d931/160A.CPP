/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem from Codeforces (https://codeforces.com/problemset/problem/160/A).
 * @details The goal is to find the minimum number of coins to take from a set so that
 * the sum of the chosen coins is strictly greater than the sum of the remaining coins.
 *
 * @algorithm Greedy Algorithm with Custom Sort
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in descending order. This is a greedy choice, as taking the
 *    largest coins first will lead to the required sum with the fewest coins.
 * 3. Iterate through the sorted coins, one by one. In each step, add the coin's
 *    value to a personal sum and subtract it from the total sum.
 * 4. Count the number of coins taken.
 * 5. Stop when the personal sum becomes strictly greater than the remaining total sum.
 * 6. The final count is the minimum number of coins required.
 *
 * @complexity
 * - Time: O(N^2), dominated by the inefficient custom `sortdesc` function.
 *   A standard library sort would reduce this to O(N log N).
 * - Space: O(N), to store the coin values in an array.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

/**
 * @brief Swaps the values of two integers using pointers.
 * @param xp Pointer to the first integer.
 * @param xy Pointer to the second integer.
 */
void swapArr(int *xp, int *xy) {
  int temp = *xp;
  *xp = *xy;
  *xy = temp;
}

/**
 * @brief Sorts an array in descending order using a naive O(N^2) method.
 * @details This function attempts to sort the array. Note: The implementation is
 * a flawed and inefficient variant of bubble/selection sort. For production code,
 * `std::sort` with a custom comparator is highly recommended.
 * @param arr The array to be sorted.
 * @param n The size of the array.
 */
void sortdesc(int arr[], int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n - 1; j++) {
      // This logic repeatedly finds the largest element and swaps it towards
      // the beginning, but it's not an optimal sorting implementation.
      if (arr[i] > arr[j]) {
        swapArr(&arr[i], &arr[j]);
      }
    }
  }
}

int main(void) {
  // n: the number of coins.
  int n;
  cin >> n;

  // C-style array to store the value of each coin.
  // Note: This is a Variable Length Array (VLA), a non-standard C++ extension.
  int coinArr[n];
  // `sum` will hold the total value of all coins.
  int sum = 0;

  /**
   * @brief Read coin values and calculate their total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> coinArr[i];
    sum += coinArr[i];
  }

  // Sort the coins in descending order to apply the greedy strategy.
  sortdesc(coinArr, n);

  // `sumMe` will accumulate the value of coins taken.
  int sumMe = 0;
  // `takeCoin` counts the number of coins taken.
  int takeCoin = 0;

  /**
   * @brief Greedily take the largest coins until the sum is sufficient.
   * This loop iterates through the descending-sorted coins, accumulating them
   * until the collected sum (`sumMe`) is strictly greater than the remaining
   * sum (`sum`).
   */
  for (int i = 0; i < n; i++) {
    // Pre-condition: Continue taking coins as long as our sum is not strictly
    // greater than the remaining sum.
    if (sumMe <= sum) {
      // Take the current largest coin.
      sumMe += coinArr[i];
      // Subtract it from the remaining sum.
      sum -= coinArr[i];
      // Increment the count of coins taken.
      takeCoin++;
    }
  }
  
  // Output the minimum number of coins required.
  cout << takeCoin;

  return 0; // Indicate successful execution.
}
