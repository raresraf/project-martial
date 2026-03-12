// https://codeforces.com/problemset/problem/160/A
/**
 * @5c6b8de6-fcad-4374-95bd-a6eeee82d931/160A.CPP
 * @brief This program solves Codeforces problem 160A, "Twins".
 * Given 'n' coin values, the objective is to find the minimum number of coins
 * that can be taken such that their sum is strictly greater than the sum of
 * the remaining coins.
 *
 * Algorithm: Greedy approach. Sorts coins in descending order and iteratively
 * takes the largest coins until their sum exceeds half of the total sum.
 * Time Complexity: O(N^2) due to the custom `sortdesc` (bubble sort equivalent).
 *                  If using `std::sort`, it would be O(N log N).
 * Space Complexity: O(N) for storing coin values.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers.
#include <iostream>      // Explicitly included for iostream.

using namespace std; // Uses the standard namespace.

/**
 * @brief Swaps the values of two integers pointed to by `xp` and `xy`.
 * @param xp Pointer to the first integer.
 * @param xy Pointer to the second integer.
 */
void swapArr(int *xp, int *xy) {
  int temp = *xp; // Inline: Stores the value of *xp temporarily.
  *xp = *xy;      // Inline: Assigns the value of *xy to *xp.
  *xy = temp;     // Inline: Assigns the temporary value to *xy.
}

/**
 * @brief Sorts an integer array in descending order using a bubble sort-like algorithm.
 * @param arr The array of integers to be sorted.
 * @param n The number of elements in the array.
 * Time Complexity: O(N^2)
 */
void sortdesc(int arr[], int n) {
  for (int i = 0; i < n; i++) { // Outer loop for passes.
    for (int j = 0; j < n - 1; j++) { // Inner loop for comparisons and swaps.
      // Inline: Compares arr[i] with arr[j]. If arr[i] is larger, it's swapped towards the beginning.
      // This is a variation of selection sort or bubble sort, leading to descending order.
      if (arr[i] > arr[j]) { 
        swapArr(&arr[i], &arr[j]);
      }
    }
  }
}

int main(void) {
  int n; // Declares 'n' for the number of coins.
  cin >> n; // Reads 'n' from standard input.

  int coinArr[n]; // Declares an array to store the coin values.
  int sum = 0;    // Initializes 'sum' to 0, which will store the total sum of all coins.

  /**
   * Block Logic: Reads 'n' coin values into `coinArr` and calculates their total sum.
   * Invariant: After this loop, `coinArr` is populated and `sum` holds the total value of all coins.
   */
  for (int i = 0; i < n; i++) {
    cin >> coinArr[i]; // Reads each coin value.
    sum += coinArr[i]; // Adds the coin value to the total sum.
  }

  sortdesc(coinArr, n); // Sorts the `coinArr` in descending order.

  int sumMe = 0;    // Initializes 'sumMe' to 0, which will store the sum of coins taken by current player.
  int takeCoin = 0; // Initializes 'takeCoin' to 0, which will count the number of coins taken.

  /**
   * Block Logic: Iterates through the coins (from largest to smallest) and takes
   * them until the sum of taken coins (`sumMe`) is strictly greater than the
   * sum of the remaining coins (`sum`).
   * Invariant: `takeCoin` counts the minimum number of coins needed to satisfy the condition.
   */
  for (int i = 0; i < n; i++) {
    // Inline: Condition to continue taking coins: if sumMe is not yet strictly greater than sum.
    if (sumMe <= sum) { 
      sumMe += coinArr[i];  // Adds the current largest coin to sumMe.
      sum -= coinArr[i];    // Subtracts the current coin from the remaining sum.
      takeCoin++;           // Increments the count of coins taken.
    }
  }
  cout << takeCoin; // Prints the minimum number of coins taken.
  // The program implicitly returns 0 on successful completion of main.
}