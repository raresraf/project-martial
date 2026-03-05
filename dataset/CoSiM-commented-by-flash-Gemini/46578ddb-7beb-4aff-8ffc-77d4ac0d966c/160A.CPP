/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem (Codeforces 160A).
 *
 * This program solves a problem where two twins want to divide a set of coins
 * such that one twin's sum of coins is strictly greater than the other's.
 * The goal is to find the minimum number of coins a twin needs to take
 * to achieve this condition.
 *
 * Algorithm: Greedy approach. Sort the coins in ascending order. Then,
 * iteratively pick the largest coins (from the end of the sorted array)
 * until their accumulated sum strictly exceeds half of the total sum of all coins.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) to store the coins.
 */
// Written By Deepesh Nimma
// Problem: 160A Codeforces
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define space " "
#define end1 "\n"
#define ar array

int n, i, a[100], sum = 0, ans = 0, cnt = 0;

int main() {
  /**
   * @brief Main function to find the minimum number of coins for one twin
   *        to have a sum strictly greater than half the total.
   *
   * Reads the number of coins (n) and their values from standard input.
   * It then applies a greedy algorithm by sorting the coins and picking
   * the largest ones until the sum exceeds half of the total.
   *
   * @return void (standard competitive programming practice).
   */
  cin >> n;

  /**
   * @brief Block Logic: Reads coin values.
   *
   * Pre-condition: 'n' is the number of coins to be read.
   * Invariant: 'a' stores the value of each coin.
   */
  for (i = 0; i < n; ++i) {
    cin >> a[i];
  }

  // Sort the 'a' array in ascending order.
  sort(a, a + n);

  /**
   * @brief Block Logic: Calculates the total sum of all coins.
   *
   * Pre-condition: 'a' contains all coin values.
   * Invariant: 'sum' accumulates the value of each coin.
   */
  for (i = 0; i < n; ++i) {
    sum += a[i];
  }

  // Calculate half of the total sum. The goal is to exceed this value.
  sum = sum / 2;

  /**
   * @brief Block Logic: Greedily selects coins starting from the largest
   *        until their sum is strictly greater than half of the total sum.
   *
   * Iterates through the sorted array from the largest coin to the smallest
   * (accessed using 'n - cnt' index).
   * Pre-condition: 'a' is sorted in ascending order.
   * Invariant: 'ans' accumulates the sum of selected coins, and 'cnt' counts them.
   */
  while (ans <= sum) {
    ++cnt;
    ans += a[n - cnt];
  }

  // Print the minimum number of coins required.
  cout << cnt;
}
