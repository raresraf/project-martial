/**
 * @file 160A.CPP
 * @brief This program solves the "Twins" problem. Given a set of coins, it determines
 * the minimum number of coins that need to be taken such that the sum of the taken
 * coins is strictly greater than the sum of the remaining coins.
 * Algorithm: Greedy approach. All coins are summed, then sorted. The largest coins are
 * iteratively taken until their sum exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) primarily due to the sorting step, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values in an array.
 */
#include <algorithm> // Required for std::sort.
#include <cstdio>    // Required for printf.
#include <fstream>   // Included in original, though not explicitly used.
#include <iostream>  // Required for std::cin.
#include <map>       // Included in original, though not explicitly used.
#include <math.h>    // Included in original, though not explicitly used.
#include <queue>     // Included in original, though not explicitly used.
#include <set>       // Included in original, though not explicitly used.
#include <stdio.h>   // Included in original, though not explicitly used (redundant with cstdio).
#include <stdlib.h>  // Included in original, though not explicitly used.
#include <streambuf> // Included in original, though not explicitly used.
#include <string.h>  // Included in original, though not explicitly used.
#include <string>    // Included in original, though not explicitly used.
#include <vector>    // Included in original, though not explicitly used.
using namespace std;

long long sum = 0;   // Global variable to store the total sum of all coin values.
int num[999999];     // Global array to store coin values. Max size 999999.

int main() {
  long long t; // Used as 'num_coins' in this context, representing the total number of coins.
  // Declared but unused local variables from original code.
  long long x, y, n, m, count = 0, a, b;

  // Reads the total number of coins.
  cin >> t;
  // Commented-out block from original, likely for multiple test cases if 't' was number of test cases.
  //    for(long long k=1;k<=t;k++)
  //    {
  sum = 0; // Resets global sum for current test case.
  long long coin = 0; // Counts the number of coins taken.
  long long val = 0;  // Stores the sum of coins currently taken.

  /**
   * Block Logic: Reads the values of 't' coins from standard input and calculates their total sum.
   * Each coin value is stored in the global 'num' array.
   * Invariant: After this loop, 'num' array contains all coin values and 'sum' holds their total.
   */
  for (long long i = 0; i < t; i++) {
    cin >> num[i];      // Reads each coin's value.
    sum += num[i];      // Adds the coin's value to the global total sum.
  }

  // Sorts the 'num' array (containing coin values) in ascending order.
  sort(num, num + t);

  /**
   * Block Logic: Iterates through the sorted coins from largest to smallest (by traversing
   * the 'num' array in reverse). It greedily adds coins to 'val' (sum of taken coins) and
   * increments 'coin' (count of taken coins) until 'val' is strictly greater than
   * 'sum' (sum of remaining coins).
   * Invariant: 'val' represents the sum of 'coin' largest coins taken so far.
   */
  for (int i = t - 1; i >= 0 && val <= sum; i--) {
    coin++;           // Increment the count of coins taken.
    val += num[i];    // Add the current largest coin to the 'val' (sum of taken coins).
    sum -= num[i];    // Subtract the taken coin from 'sum' (which now represents remaining sum).
  }
  // Prints the minimum number of coins required to satisfy the condition, followed by a newline.
  printf("%lld\n", coin);
  // Commented-out closing brace for the original 'for' loop.
  //    }
  // Standard program exit.
  return 0;
}
