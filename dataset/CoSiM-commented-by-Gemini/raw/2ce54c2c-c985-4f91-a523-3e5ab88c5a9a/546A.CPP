/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow to buy a specific number of bananas,
 * where each banana has a sequentially increasing cost.
 * The problem is available at: https://codeforces.com/problemset/problem/546/A
 *
 * Algorithm:
 * The cost of the bananas forms an arithmetic progression. The first banana costs `k`, the second `2k`, up to the `w`-th banana costing `w*k`.
 * The total cost is the sum of this series: `k * (1 + 2 + ... + w)`.
 * The program calculates this total cost iteratively.
 * 1. Read the cost of the first banana `k`, the initial money `n`, and the number of bananas to buy `w`.
 * 2. Initialize a `sum` variable to 0.
 * 3. Loop from `i = 1` to `w`, adding `i * k` to the `sum` in each iteration.
 * 4. After the loop, the `sum` holds the total cost.
 * 5. If the `sum` is greater than the initial money `n`, the amount to borrow is `sum - n`.
 * 6. Otherwise, no money is needed, so the amount to borrow is 0.
 * The result is printed to the console.
 *
 * Time Complexity: O(w), due to the loop that iterates 'w' times to calculate the sum.
 * Space Complexity: O(1), using a fixed number of variables.
 */
// https://codeforces.com/problemset/problem/546/A

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

using namespace std;

int main() {
  // k: cost of the first banana.
  // n: initial amount of money the soldier has.
  // w: number of bananas to buy.
  int k, n, w;
  cin >> k >> n >> w;

  int sum = 0;
  /**
   * Block Logic: This loop calculates the total cost of buying 'w' bananas.
   * Pre-condition: 'w' is the number of bananas, 'k' is the base cost.
   * Invariant: At the end of each iteration 'i', 'sum' holds the total cost of the first 'i' bananas.
   */
  for (int i = 1; i <= w; i++) {
    sum += i * k;
  }

  /**
   * Block Logic: Determine the amount of money to borrow.
   * If the total cost 'sum' exceeds the soldier's initial money 'n', the difference is the required loan.
   * Otherwise, no loan is needed.
   */
  if (sum > n)
    cout << sum - n;
  else
    cout << "0";
    
  return 0;
}
