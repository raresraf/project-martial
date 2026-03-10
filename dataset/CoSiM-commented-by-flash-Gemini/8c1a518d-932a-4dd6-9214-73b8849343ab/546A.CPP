/**
 * @file 546A.CPP
 * @brief This program solves the "Soldier and Bananas" problem (Codeforces 546A).
 *   A soldier wants to buy `w` bananas. The first costs `k` dollars, the second `2k`, and so on.
 *   He has `n` dollars. The task is to calculate how much money he needs to borrow.
 *   If he has enough money, the amount to borrow is 0.
 *   Problem description: https://codeforces.com/problemset/problem/546/A
 * * Algorithm: Iterative sum and conditional calculation.
 *   1. It calculates the sum `1 + 2 + ... + w` using a loop.
 *   2. The total cost is then derived by multiplying this sum by `k`.
 *   3. The amount to borrow is the total cost minus the money the soldier has, clamped at a minimum of 0.
 * Time Complexity: O(W) where W is the number of bananas, due to the loop for summing `1` to `w`.
 *   For typical competitive programming constraints (W up to 1000), this is efficient enough.
 * Space Complexity: O(1) for a few integer/long long variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

/**
 * @brief Function to solve the Soldier and Bananas problem.
 * Functional Utility: Reads `k`, `n`, `w` from standard input, calculates the total cost
 *                     of bananas, determines the amount to borrow, and prints the result.
 */
void solve() {
  int k, w;    // `k`: cost of the first banana, `w`: number of bananas to buy.
  long long n; // `n`: money soldier has.
  // Block Logic: Reads the input values for k, n, and w.
  cin >> k >> n >> w;
  int s = 0;   // `s`: Stores the sum of `1 + 2 + ... + w`.
  /**
   * Block Logic: Calculates the sum `1 + 2 + ... + w` iteratively.
   * Pre-condition: `w` is a positive integer.
   * Invariant: `s` accumulates the sum of integers from 1 up to the current loop variable `i`.
   */
  for (int i = 1; i <= w; i++) {
    s += i;
  }
  // `kq`: Stores the calculated amount to borrow.
  // Functional Utility: Calculates the total cost (`k * s`) and then subtracts the money the soldier has (`n`).
  long long kq = k * s - n;
  /**
   * Conditional Logic: Determines whether the soldier needs to borrow money.
   * If `kq` is greater than 0, it means he needs to borrow that amount.
   * Otherwise, he has enough money or more, so he needs to borrow 0.
   */
  if (kq > 0)
    cout << kq; // Prints the positive amount to borrow.
  else
    cout << 0;  // Prints 0 if no money needs to be borrowed.
}
/**
 * @brief Main function to initiate the problem-solving process.
 * Functional Utility: Calls the `solve()` function to handle the problem logic.
 */
int main() {
  solve();
  return 0;
}
