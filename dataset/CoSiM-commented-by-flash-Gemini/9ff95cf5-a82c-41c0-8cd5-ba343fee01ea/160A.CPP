/**
 * @9ff95cf5-a82c-41c0-8cd5-ba343fee01ea/160A.CPP
 * @brief Solution for a competitive programming problem to find the minimum coins for a sum greater than half the total.
 *
 * This program determines the minimum number of coins required to achieve a sum
 * that is strictly greater than half of the total sum of all available coins.
 * It employs a greedy strategy by always picking the largest available coins first.
 *
 * Algorithm:
 * 1. Read the number of coins 'n'.
 * 2. Read the values of all 'n' coins into an array and calculate their total sum.
 * 3. Sort the coins in non-decreasing order.
 * 4. Calculate the target sum: half of the total sum.
 * 5. Iterate from the largest coin downwards, accumulating their values and counting
 *    the number of coins taken, until the accumulated sum exceeds the target sum.
 * 6. Print the count of coins taken.
 *
 * Time Complexity: O(N log N) primarily due to the sorting step.
 * Space Complexity: O(N) to store the coin values in an array.
 */

#include <algorithm> // Functional Utility: Provides the std::sort algorithm for sorting elements.
#include <iostream>  // Functional Utility: Provides input/output functionalities (std::cin, std::cout).
using namespace std;

int main() {
  int n;
  // Functional Utility: Reads the total number of coins.
  cin >> n;
  // Functional Utility: Declares an array to store coin values. Max size 105.
  int a[105];
  // Functional Utility: Variable to store the total sum of all coin values.
  int sm = 0;
  // Block Logic: Reads coin values and calculates their total sum.
  // Pre-condition: 'n' integer values representing coin values are available in input.
  // Invariant: 'a' will contain all coin values, and 'sm' will hold their sum.
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
    sm += a[i];
  }

  // Functional Utility: Sorts the array of coin values in non-decreasing order.
  sort(a, a + n);

  // Functional Utility: Calculates half of the total sum. Using float to ensure precision for division.
  float mid = sm / 2.0;
  // Functional Utility: Variable to accumulate the sum of taken coins.
  int sum = 0;
  // Functional Utility: Index to iterate from the largest coin (end of sorted array).
  int x = n - 1;
  // Functional Utility: Counter for the number of coins taken.
  int coins = 0;
  // Block Logic: Greedily takes coins from largest to smallest until their sum exceeds 'mid'.
  // Pre-condition: Coins are sorted, and 'mid' is the target sum.
  // Invariant: Loop continues as long as the accumulated sum is not sufficient and there are coins left.
  while (sum <= mid && x >= 0) {
    sum += a[x];  // Functional Utility: Adds the value of the current largest coin.
    x--;          // Functional Utility: Moves to the next largest coin.
    coins++;      // Functional Utility: Increments the count of taken coins.
  }

  // Functional Utility: Prints the minimum number of coins required, followed by a newline.
  cout << coins << "\n";
}
