/**
 * @file 546A.CPP
 * @brief Solution for Codeforces problem 546A, "Soldier and Bananas".
 *
 * @description
 * A soldier buys `w` bananas. The cost increases linearly: the first banana
 * costs `k`, the second `2k`, up to the `w`-th banana costing `w*k`. Given the
 * soldier's initial money `n`, this program calculates how much money the
 * soldier must borrow. If the soldier has enough, the amount is 0.
 *
 * @algorithm
 * 1. Read the initial cost `k`, the soldier's money `n`, and the number of
 *    bananas `w`.
 * 2. Iteratively calculate the total cost by summing the price of each banana
 *    from 1 to `w`.
 * 3. Compare the total cost with the money `n`.
 * 4. If the cost is greater than `n`, the amount to borrow is the difference.
 * 5. Otherwise, the amount to borrow is 0.
 * 6. Print the amount to borrow.
 *
 * @time_complexity O(W), where W is the number of bananas.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // k: base cost of a banana.
  // w: number of bananas to buy.
  // n: soldier's initial money.
  int k, w;
  long long int n;

  // cost: accumulator for the total price of all bananas.
  int cost = 0;
  // borrow: the amount of money the soldier needs to borrow.
  long long int borrow = 0;

  cin >> k >> n >> w;

  // Block Logic: Sum the cost of each banana from the 1st to the w-th.
  for (int i = 1; i <= w; i++) {
    cost = cost + k * i;
  }

  // Block Logic: Determine if the soldier needs to borrow money.
  if (n < cost) {
    // If money is insufficient, calculate the deficit.
    borrow = cost - n;
    cout << borrow << endl;
  } else {
    // If money is sufficient, the amount to borrow is 0.
    cout << borrow << endl;
  }
  return 0;
}
