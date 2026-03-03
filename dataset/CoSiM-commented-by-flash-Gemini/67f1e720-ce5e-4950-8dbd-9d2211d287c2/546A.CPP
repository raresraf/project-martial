/**
 * @file 546A.CPP
 * @brief Solution for Codeforces Problem 546A - "Soldier and Bananas".
 * This program calculates the amount of money a soldier needs to borrow to buy a certain
 * number of bananas, given the cost progression and the money the soldier already has.
 * The cost of the i-th banana is i*k.
 * Algorithm: Iterative summation of banana costs.
 * Time Complexity: O(W), where W is the number of bananas.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries (e.g., iostream, etc.).
using namespace std;     // Uses the standard namespace.

int main() {
  // k: the cost of the first banana.
  // n: the amount of money the soldier has.
  // w: the number of bananas the soldier wants to buy.
  // cost: accumulates the total cost of buying 'w' bananas.
  int k, n, w, cost = 0;
  
  // Reads the input values: k (first banana cost), n (money soldier has), w (number of bananas).
  cin >> k >> n >> w;
  
  // num: a copy of 'w' for loop control.
  // i: represents the current banana number (1-indexed).
  int num = w, i = 1;

  /**
   * Block Logic: Iteratively calculates the total cost of buying 'w' bananas.
   * In each step, the cost of the current banana (i*k) is added to the total.
   * Invariant: `cost` holds the cumulative cost of bananas bought up to the current iteration.
   */
  while (num > 0) { // Loop 'w' times.
    cost += i * k; // Add the cost of the i-th banana to the total.
    i++;           // Increment to the next banana number.
    num--;         // Decrement the count of remaining bananas to buy.
  }

  // Calculates the amount the soldier needs to borrow.
  int borrow = cost - n;
  
  // Conditional output: If the total cost is less than or equal to the money the soldier has,
  // output 0 (no borrowing needed). Otherwise, output the calculated borrowing amount.
  if (cost <= n)
    cout << 0;
  else
    cout << borrow;
    
  return 0; // Indicates successful program execution.
}
