#include <bits/stdc++.h>
using namespace std;

/**
 * @file 546A.CPP
 * @brief Solution for a problem involving a simple arithmetic progression,
 * often called "Soldier and Bananas".
 *
 * Problem Statement:
 * A soldier wants to buy `w` bananas. The first banana costs `k`, the second
 * `2k`, the third `3k`, and so on. The soldier has `n` dollars. The task is to
 * calculate the amount of money the soldier must borrow to buy all `w` bananas.
 * If the soldier has enough money, the amount to borrow is 0.
 *
 * Algorithm:
 * The program iteratively calculates the total cost of all bananas.
 * 1. Read the initial cost `k`, the soldier's initial money `n`, and the
 *    number of bananas to buy `w`.
 * 2. Loop from 1 to `w`, representing each banana. In each iteration `i`, add
 *    the cost of the `i`-th banana (`i * k`) to a running total.
 * 3. After the loop, compare the total cost with the soldier's money.
 * 4. If the money is sufficient (greater than or equal to the total cost),
 *    the required loan is 0.
 * 5. Otherwise, the loan is the difference between the total cost and the money.
 *
 * @note A more computationally efficient O(1) solution exists using the formula
 * for the sum of an arithmetic series: Total Cost = k * (w * (w + 1) / 2).
 *
 * Time Complexity: O(W), where W is the number of bananas to buy.
 * Space Complexity: O(1).
 */

int main() {

  // cost: The cost of the first banana (k).
  // mon: The initial amount of money the soldier has (n).
  // need: The number of bananas the soldier wants to buy (w).
  // tot: The running total cost of the bananas.
  long long int cost, mon, need, tot = 0;
  cin >> cost >> mon >> need;

  // Block Logic: Calculate the total cost by summing the price of each banana.
  // Invariant: 'tot' stores the cost of the first 'i-1' bananas.
  for (int i = 1; i <= need; i++) {
    // The cost of the i-th banana is i * cost.
    tot += i * cost;
  }

  // Pre-condition: Check if the soldier has enough money.
  if (tot <= mon) {
    // If the total cost is less than or equal to the money available, no loan is needed.
    cout << "0" << endl;
  } else {
    // Otherwise, calculate the deficit.
    cout << tot - mon << endl;
  }

  return 0;
}