/**
 * @file 546A.CPP
 * @brief Solution for a competitive programming problem involving calculating money needed to borrow.
 * Algorithm: Iterative summation of costs with conditional output based on total cost versus available funds.
 * Time Complexity: $O(W)$ where W is the number of bananas to buy.
 * Space Complexity: $O(1)$ as only a few variables are used.
 */

#include <iostream>

using namespace std;

int main() {
  // k: The cost of the first banana.
  // n: The initial amount of money the soldier has.
  // w: The number of bananas the soldier wants to buy.
  int k, n, w;
  // total: Accumulates the total cost of all bananas.
  int total = 0;
  // borrow: Stores the amount of money the soldier needs to borrow.
  int borrow = 0;

  cin >> k >> n >> w;

  // Block Logic: Calculates the total cost of buying 'w' bananas.
  // Invariant: At the start of each iteration, 'total' holds the sum of costs for (i-1) bananas.
  for (int i = 1; i <= w; ++i) {
    total += k * i;
  }

  // Block Logic: Determines if the soldier needs to borrow money and calculates the amount.
  if (n >= total) {
    cout << 0 << endl;
  } else {
    borrow = total - n;
    cout << borrow << endl;
  }

  return 0;
}