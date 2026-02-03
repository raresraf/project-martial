/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem.
 * @see https://codeforces.com/problemset/problem/160/A
 *
 * This program determines the minimum number of coins to take so that their
 * sum is strictly greater than the sum of the remaining coins.
 *
 * Algorithm:
 * 1. Read the number of coins (n) and the value of each coin.
 * 2. Calculate the total sum of all coins.
 * 3. Sort the coins in ascending order.
 * 4. Iterate through the sorted coins in reverse (from largest to smallest).
 * 5. Accumulate the sum of the taken coins (`sum2`) and count them.
 * 6. Stop when `sum2` becomes strictly greater than half of the total sum.
 * 7. Print the number of coins taken.
 *
 * Time Complexity: O(N log N) due to sorting.
 * Space Complexity: O(N) to store the coin values.
 */
#include <algorithm>
#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  int arr[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n);
  int count = 0, sum2 = 0;
  // Invariant: This loop iterates from the largest coin to the smallest,
  // accumulating a sum until it is greater than half of the total sum.
  for (int i = n - 1; i >= 0; i--) {
    // Pre-condition: If the accumulated sum is already greater than half the
    // total, no more coins are needed.
    if (sum2 > sum / 2) {
      break;
    } else {
      sum2 += arr[i];
      count++;
    }
  }
  cout << count;
  return 0;
}