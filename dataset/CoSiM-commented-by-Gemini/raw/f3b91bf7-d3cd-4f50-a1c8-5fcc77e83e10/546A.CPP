/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 * @details Calculates the amount of money a soldier needs to borrow to buy a
 * specific number of bananas with an increasing cost.
 *
 * The problem involves a soldier buying `w` bananas. The first banana costs `k`,
 * the second `2k`, and so on, up to the w-th banana costing `w*k`. The soldier
 * initially has `n` dollars. The program calculates the total cost and determines
 * how much money must be borrowed, which is the total cost minus the initial money.
 * If the soldier has enough money, the amount to borrow is 0.
 *
 * @author f3b91bf7-d3cd-4f50-a1c8-5fcc77e83e10
 *
 * @algorithm Arithmetic Series Summation
 * @complexity
 * Time: O(W), where W is the number of bananas, due to the loop.
 *       This could be O(1) by using the formula `k * W * (W + 1) / 2`.
 * Space: O(1).
 */
/*
Codeforces Round #304 (Div. 2) - A. Soldier and Bananas
https://codeforces.com/problemset/problem/546/A
*/

#include <bits/stdc++.h>
using namespace std;

// Macro for fast I/O in C++.
#define FAST_INP                                                               
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL)

int main() {
  FAST_INP;
  // `k`: cost of the first banana.
  // `n`: initial amount of money the soldier has.
  // `w`: total number of bananas to buy.
  int k, n, w;
  cin >> k >> n >> w;
  
  // `t`: accumulator for the total cost.
  int t = 0;
  
  /**
   * @brief Loop to calculate the total cost of `w` bananas.
   * The cost of the i-th banana is `i * k`.
   */
  for (int i = 1; i <= w; i++)
    t += i * k;
    
  /**
   * @brief Calculate and print the amount to borrow.
   * If the initial money `n` is greater than or equal to the total cost `t`,
   * the soldier borrows 0. Otherwise, the amount to borrow is `t - n`.
   * This is handled concisely with a ternary operator.
   */
  cout << (n >= t ? 0 : t - n) << "
";
  
  return 0;
}