/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem from Codeforces (Problem 546A).
 * Problem Statement: A soldier wants to buy `w` bananas. The first banana costs `k` dollars,
 *                    the second `2k`, and so on. He has `n` dollars. Determine how much money
 *                    he needs to borrow to buy all `w` bananas.
 * Algorithm: Direct summation. Calculates the total cost of `w` bananas and subtracts the
 *            money the soldier already has. If the result is negative or zero, no money needs to be borrowed.
 * Time Complexity: O(w) for the loop to calculate total cost.
 * Space Complexity: O(1) for constant extra variables.
 * Domain: Competitive Programming, Simple Arithmetic.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to solve the Soldier and Bananas problem.
 * Functional Utility: Calculates the total cost of purchasing a specified number of bananas
 *                     and determines the amount of money a soldier needs to borrow.
 */
int main() {
  // Functional Utility: Optimizes C++ standard input/output operations.
  // Logic: Disables synchronization with C's stdio and unties cin from cout for faster execution.
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  // Variables:
  // k: Cost of the first banana.
  // n: Amount of money the soldier has.
  // w: Number of bananas the soldier wants to buy.
  // money: Total cost of all bananas.
  // ans: The amount of money the soldier needs to borrow.
  long long k, n, w, money = 0, ans;
  // Block Logic: Reads the cost of the first banana (k), the money the soldier has (n),
  //              and the number of bananas to buy (w).
  cin >> k;
  cin >> n;
  cin >> w;
  // Block Logic: Calculates the total cost of `w` bananas.
  // Invariant: The i-th banana costs `i * k`. `money` accumulates this sum.
  for (int i = 1; i <= w; i++) { // Loop from 1 to w, as the 1st banana costs 1*k.
    money = money + i * k;
  }
  // Block Logic: Determines if the soldier needs to borrow money.
  // If total cost (`money`) is greater than what he has (`n`), he needs to borrow `money - n`.
  if (money > n) {
    ans = money - n;
    cout << ans;
  }
  // Otherwise, he has enough money or more, so he needs to borrow 0.
  else {
    cout << "0";
  }
  return 0;
}
