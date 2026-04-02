/**
 * @file 546A.CPP
 * @brief Solution to a computational problem involving calculating the cost of a series of items.
 *
 * Algorithm:
 * This program calculates the total cost of buying 'w' items where the price increases
 * arithmetically. The first item costs 'k', the second '2k', and so on, up to the
 * 'w'-th item costing 'w*k'. It then determines the amount of money that needs to be
 * borrowed, which is the total cost minus the initial amount of money 'n'.
 * The total cost is calculated by summing an arithmetic progression.
 *
 * Time Complexity: O(w) - The cost is calculated by iterating 'w' times.
 *   Note: This could be optimized to O(1) by using the formula for the sum of an
 *   arithmetic series: total_cost = k * w * (w + 1) / 2.
 *
 * Space Complexity: O(1) - The program uses a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // Fast I/O operations
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  
  // k: cost of the first item
  // n: initial amount of money
  // w: number of items to buy
  long long k, n, w, money = 0, ans;
  cin >> k;
  cin >> n;
  cin >> w;
  
  /**
   * Block Logic: Calculate the total cost of buying 'w' items.
   * Pre-condition: k, n, w are initialized with input values.
   * Invariant: 'money' accumulates the total cost.
   * The cost of the i-th item is i * k. This loop sums the costs for i from 0 to w.
   */
  for (int i = 0; i <= w; i++) {
    money = money + i * k;
  }
  
  /**
   * Block Logic: Determine the amount to borrow.
   * If the total cost 'money' is greater than the initial amount 'n',
   * calculate the difference. Otherwise, no money needs to be borrowed.
   */
  if (money > n) {
    ans = money - n;
    cout << ans;
  } else {
    cout << "0";
  }
  return 0;
}
