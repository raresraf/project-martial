/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/546/A
 *
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the bananas increases linearly.
 */
/*
 *Link: https://codeforces.com/problemset/problem/546/A
 */

#include <iostream>
using namespace std;

int main() {
  // k: cost of the first banana
  // n: initial money
  // w: number of bananas to buy
  long long k, n, w, total_price, res;
  cin >> k >> n >> w;
  
  /**
   * @brief Calculate the total cost of w bananas.
   * The cost is an arithmetic progression: k, 2k, 3k, ..., wk.
   * The sum is k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
   */
  total_price = k * ((w * (w + 1)) / 2);
  
  // Calculate the amount to borrow.
  res = total_price - n;
  
  /**
   * @brief If the soldier has enough money, they don't need to borrow anything.
   */
  if (res <= 0) {
    res = 0;
  }
  cout << res << endl;
}
