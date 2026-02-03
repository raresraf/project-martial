/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem.
 * @see https://codeforces.com/problemset/problem/546/A
 *
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the bananas increases linearly.
 *
 * Algorithm:
 * 1. Read the cost of the first banana (k), the initial amount of money the
 *    soldier has (n), and the number of bananas he wants to buy (w).
 * 2. Calculate the total cost of w bananas. The cost of the i-th banana is
 *    i * k. The total cost is the sum of an arithmetic progression:
 *    k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
 * 3. If the total cost is greater than the money the soldier has, print the
 *    difference. Otherwise, print 0.
 *
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  long long int k, n, w, w1;
  cin >> k >> n >> w;
  // Functional utility: Calculates the total cost of `w` bananas using the
  // formula for the sum of an arithmetic series.
  w1 = (k * w * w + k * w) / 2;
  // Block Logic: Determines if the soldier needs to borrow money.
  // If the total cost `w1` is greater than the initial money `n`, the
  // difference is the amount to borrow. Otherwise, no money needs to be
  // borrowed.
  if (w1 > n) {
    cout << w1 - n << endl;
  } else
    cout << "0" << endl;
}