/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem (Codeforces 546A).
 * @details This program calculates the amount of money a soldier needs to borrow to buy a
 * specific number of bananas. The cost of each subsequent banana increases arithmetically.
 *
 * Algorithm: Arithmetic Series Summation
 * The cost of 'w' bananas forms an arithmetic series: k*1 + k*2 + ... + k*w.
 * This can be simplified to k * (1 + 2 + ... + w).
 * The program uses the formula for the sum of the first 'w' positive integers, which is (w * (w + 1)) / 2.
 * It calculates the total cost, and if the cost exceeds the soldier's initial money,
 * it computes the difference. Otherwise, the amount to borrow is zero.
 *
 * Time Complexity: O(1), as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1), using a constant amount of memory.
 */

#include <iostream>
using namespace std;
int main() {
  // k: the cost of the first banana.
  // n: the initial amount of money the soldier has.
  // w: the number of bananas the soldier wants to buy.
  // m: will store the total cost of 'w' bananas.
  long long int n, w, k, m = 0;

  // Read the input values.
  cin >> k;
  cin >> n;
  cin >> w;
  
  // Block Logic: Calculate the total cost of 'w' bananas.
  // First, calculate the sum of the arithmetic series 1 + 2 + ... + w.
  m = (w * (w + 1)) / 2;
  // Then, multiply by the base cost 'k' to get the total cost.
  m = m * k;
  
  // Block Logic: Determine the amount of money to borrow.
  // Pre-condition: 'm' is the total cost, 'n' is the money available.
  if (n <= m)
    // If the total cost is greater than or equal to the money the soldier has,
    // the amount to borrow is the difference. If m == n, this is 0.
    cout << m - n;
  else
    // If the soldier has more money than required, they need to borrow nothing.
    cout << 0;
}
