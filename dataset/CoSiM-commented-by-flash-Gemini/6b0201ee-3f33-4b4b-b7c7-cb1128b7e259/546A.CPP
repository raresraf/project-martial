/**
 * @file 546A.CPP
 * @brief Solution to a competitive programming problem: "Soldier and Bananas".
 * Calculates the amount of money a soldier needs to borrow to buy 'w' bananas.
 * The first banana costs 'k', the second '2k', and so on.
 * Algorithm: Direct calculation based on arithmetic series sum.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

/**
 * @brief Main function to solve the Soldier and Bananas problem.
 * Reads the cost of the first banana (k), initial money (n), and number of bananas (w).
 * Calculates the total cost of bananas and outputs the amount of money to borrow.
 */
int main() {
  ll k, n, w;
  cin >> k >> n >> w;
  ll ok = (w * (w + 1)) / 2;
  ll res = ok * k;
  /**
   * Block Logic: Determines if the soldier needs to borrow money.
   * If the total cost (res) is greater than initial money (n),
   * calculates the deficit; otherwise, no money needs to be borrowed.
   */
  if (n < res)
    cout << res - n << endl;
  else
    cout << "0" << endl;
}

