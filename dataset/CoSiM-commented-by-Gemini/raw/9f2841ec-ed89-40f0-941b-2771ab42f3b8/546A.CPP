/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 *
 * This program calculates the amount of money a soldier needs to borrow to buy
 * a certain number of bananas. The cost of the bananas increases linearly.
 *
 * Algorithm: Mathematical formula. The total cost of 'w' bananas, where the
 * i-th banana costs i*k, is calculated using the formula for the sum of an
 * arithmetic series. Total cost = k * (1 + 2 + ... + w) = k * w * (w + 1) / 2.
 * The amount to borrow is the difference between the total cost and the money
 * the soldier has ('n'). If the cost is less than 'n', no money needs to be
 * borrowed, so the result is 0.
 *
 * Time Complexity: O(1), as it's a direct calculation.
 * Space Complexity: O(1).
 */
#include <iostream>
#include <algorithm> // Required for std::max

using namespace std;

int main() {
  long long k; // The cost of the first banana.
  long long n; // The initial amount of money the soldier has.
  long long w; // The number of bananas the soldier wants to buy.
  
  cin >> k >> n >> w;

  // Calculate the total cost of 'w' bananas.
  long long total_cost = w * (w + 1) / 2 * k;
  
  // Calculate the amount to borrow. If the cost is less than or equal to
  // the money on hand, the result is 0.
  long long borrow_amount = max(0LL, total_cost - n);
  
  cout << borrow_amount;
  
  return 0;
}
