/**
 * @file 546A.CPP
 * @brief Solution for the "Soldier and Bananas" problem.
 *
 * @problem A soldier wants to buy 'w' bananas. The first banana costs 'k', the second
 * costs '2k', and so on. The soldier has 'n' dollars. The task is to calculate
 * how much money the soldier needs to borrow. If he has enough money, he borrows 0.
 *
 * @algorithm The problem can be solved with a direct mathematical formula.
 * 1. Calculate the total cost of 'w' bananas. This is an arithmetic series:
 *    Cost = k + 2k + 3k + ... + w*k
 *         = k * (1 + 2 + 3 + ... + w)
 *    Using the formula for the sum of the first 'w' integers, `w*(w+1)/2`, the
 *    total cost is `(k * w * (w + 1)) / 2`.
 * 2. Calculate the amount to borrow, which is `Total Cost - n`.
 * 3. If the result is positive, that's the answer. Otherwise, the answer is 0.
 *
 * @time_complexity O(1), as the solution involves a fixed number of arithmetic operations.
 * @space_complexity O(1).
 */
#include <iostream>
using namespace std;
int main() {
  // k: cost of the first banana
  // n: initial amount of money
  // w: number of bananas to buy
  long long int k, n, w, i, j, sum = 0;
  cin >> k >> n >> w;
  
  // Calculate the total cost using the arithmetic series summation formula.
  sum = (k * w * (w + 1)) / 2;
  
  // Calculate the difference between the total cost and the money available.
  long long int ans = sum - n;

  // If the soldier has enough money (ans <= 0), he borrows 0.
  // Otherwise, he borrows the calculated difference.
  if (ans < 0)
    cout << "0";
  else
    cout << ans;
    
  return 0;
}
