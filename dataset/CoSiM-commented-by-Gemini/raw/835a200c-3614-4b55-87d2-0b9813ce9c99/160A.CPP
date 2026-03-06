/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem by finding the minimum number of coins for a majority sum.
 *
 * The problem is to determine the minimum number of coins one needs to take from a
 * collection so that the sum of the taken coins is strictly greater than the sum
 * of the remaining coins.
 *
 * @algorithm Bubble Sort followed by a Greedy selection.
 * @details The program first sorts the coins in ascending order using a manual
 * bubble sort. It then greedily takes the largest coins (by iterating through
 * the sorted array in reverse) until their sum exceeds half of the total sum.
 *
 * @complexity
 * Time: O(N^2), where N is the number of coins. The complexity is dominated by
 *       the inefficient Bubble Sort implementation. A more optimal solution
 *       using `std::sort` would be O(N log N).
 * Space: O(N) to store the coin values in an array.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to handle input and execute the algorithm.
 */
int main() {
  int n, i, temp, j;
  int sum = 0; // Used to store the total sum of all coins.
  cin >> n;
  int a[n]; // A variable-length array (VLA), which is a non-standard C++ extension.

  // --- 1. Input and Summation Phase ---
  // Block Logic: Read all coin values into the array and calculate their total sum.
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum = sum + a[i];
  }

  // The target sum to beat is half of the total sum.
  sum = sum / 2;

  // --- 2. Sorting Phase (Bubble Sort) ---
  // Pre-condition: Array 'a' is unsorted.
  // Block Logic: The following nested loops implement a Bubble Sort to arrange
  // the coins in ascending order. This is an O(N^2) operation.
  for (i = 0; i < n; i++) {
    for (j = 0; j < n - i - 1; j++) {
      if (a[j] > a[j + 1]) {
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
      }
    }
  }

  // --- 3. Greedy Selection Phase ---
  j = 0;    // Will count the number of coins taken.
  temp = 0; // Will accumulate the sum of coins taken.
  
  // Pre-condition: Array 'a' is sorted ascendingly.
  // Block Logic: Iterate backwards from the largest coin. Keep taking coins
  // until the sum of taken coins ('temp') is strictly greater than half the
  // total sum ('sum').
  for (i = n - 1; i >= 0; i--) {
    if (temp > sum)
      break;
    else {
      temp = temp + a[i];
      j++;
    }
  }
  
  // Output the minimum number of coins needed.
  cout << j;
  
  return 0;
}
