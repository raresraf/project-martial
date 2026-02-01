/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem (Codeforces 546A).
 * This program calculates the minimum amount of money a soldier needs to borrow
 * to buy 'w' bananas, given the cost 'k' of the first banana and his initial money 'n'.
 * The cost of each subsequent banana increases linearly (1st costs k, 2nd costs 2k, ..., w-th costs wk).
 * Algorithm: The total cost of 'w' bananas is calculated as the sum of an arithmetic progression:
 *   k + 2k + ... + wk = k * (1 + 2 + ... + w) = k * (w * (w + 1) / 2).
 * The program calculates this total cost by iteratively summing `i * k` in a loop.
 * The amount to borrow is then determined by `max(0, total_cost - n)`.
 * Time Complexity: O(w), due to the loop iterating 'w' times to sum the costs.
 *   A direct mathematical formula for the sum (k * w * (w + 1) / 2) would yield O(1) complexity.
 * Space Complexity: O(1), as it only stores a few integer variables.
 */
/*
 * 546A - Soldier and Bananas.cpp
 * http://codeforces.com/problemset/problem/546/A
 *  Created on: Oct 28, 2017
 *      Author: athena
 * 31848244 	546A - 23 	GNU C++14 	Accepted 	15 ms 	4 KB
 * 2017-10-28 21:46:27 	2017-10-28 21:46:27
 */

#include <iostream>
using namespace std;

int main() {
  int k, n, w;
  // Functional Utility: Reads three integers: 'k' (cost of first banana), 'n' (initial money), and 'w' (number of bananas).
  cin >> k >> n >> w;

  int totalCost = 0;
  // Block Logic: Calculates the total cost of 'w' bananas by summing the cost of each banana.
  // The cost of the i-th banana is i * k.
  // Invariant: 'totalCost' accumulates the sum of an arithmetic progression.
  for (int i = 1; i <= w; i++) {
    totalCost += i * k;
  }

  // Conditional Logic: Determines if the soldier has enough money or needs to borrow.
  // Precondition: 'totalCost' holds the total expense for 'w' bananas, 'n' is initial money.
  // If 'totalCost' is less than or equal to 'n', no borrowing is needed.
  if (totalCost <= n) {
    cout << "0" << endl;
  } else {
    // Block Logic: Calculates the amount of money the soldier needs to borrow.
    int amountToBorrow = totalCost - n;
    cout << amountToBorrow << endl;
  }

  return 0;
}
