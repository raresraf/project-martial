/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem by finding the minimum number of coins for a majority sum.
 *
 * The program aims to find the minimum number of coins one needs to take from a set
 * so that the sum of the values of the taken coins is strictly greater than the sum
 * of the values of the remaining coins. The optimal strategy is a greedy approach:
 * always take the highest-value coins first.
 *
 * @author f2c51d08-a0bb-48bc-86cc-e169abc465da
 *
 * @algorithm Greedy Algorithm with Sorting
 * @complexity
 * Time: O(T log T), dominated by the sorting of the `t` coins.
 * Space: O(T), to store the coin values in an array.
 */
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <streambuf>
#include <string.h>
#include <string>
#include <vector>
using namespace std;
// `sum`: A global variable used to track coin sums.
long long sum = 0;
// `num`: A large array to store the values of the coins.
int num[999999];
int main() {
  // `t`: stores the number of coins. Other variables are unused.
  long long t, x, y, n, m, count = 0, a, b;
  cin >> t;

  // The code has remnants of a multi-test-case structure.
  //    for(long long k=1;k<=t;k++)
  //    {
  sum = 0;
  // `coin`: counts the number of coins taken.
  // `val`: accumulates the sum of the coins taken.
  long long coin = 0, val = 0;
  /**
   * @brief First pass: Read coin values and calculate the total sum.
   */
  for (long long i = 0; i < t; i++) {
    cin >> num[i];
    sum += num[i];
  }
  // Sort the coins in non-decreasing order.
  sort(num, num + t);
  /**
   * @brief Second pass: Greedily take the largest coins.
   * This loop iterates from the largest coin downwards. It stops once the sum
   * of the taken coins (`val`) is strictly greater than the sum of the remaining
   * coins (`sum`).
   * Pre-condition: `num` is sorted, `val` is 0, `sum` is the total sum.
   * Invariant: `val` increases and `sum` decreases with each iteration. `coin`
   *            tracks the number of coins taken.
   */
  for (int i = t - 1; i >= 0 && val <= sum; i--) {
    coin++;
    val += num[i]; // Add the largest available coin to our pile.
    sum -= num[i]; // Remove it from the pile of remaining coins.
  }
  // Print the minimum number of coins required.
  printf("%lld
", coin);
  //
  //    }
  return 0;
}