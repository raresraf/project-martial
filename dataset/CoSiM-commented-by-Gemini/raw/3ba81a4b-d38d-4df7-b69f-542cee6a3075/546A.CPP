/**
 * @file 546A.CPP
 * @brief A solution to the "Soldier and Bananas" problem (Codeforces 546A)
 *        with confusing variable names.
 *
 * @details This program calculates the amount of money a soldier needs to borrow.
 * The logic is correct, but the variable names are misleadingly mapped to the
 * problem statement's inputs.
 *
 * @algorithm The program calculates the total cost of `w` bananas, which form an
 * arithmetic series.
 * 1. It reads the problem's inputs `k, n, w` into variables `n, k, w` respectively.
 *    - `code's n` = `problem's k` (cost of first banana)
 *    - `code's k` = `problem's n` (initial money)
 *    - `code's w` = `problem's w` (number of bananas)
 * 2. It calculates the total cost using the formula `n * (w * (w + 1) / 2)`,
 *    which corresponds to `k * sum(1..w)`.
 * 3. It calculates the amount to borrow (`b`) by subtracting the initial money
 *    (`k`) from the total cost (`a`).
 * 4. If the amount to borrow is negative, it outputs 0.
 *
 * @complexity
 * Time Complexity: O(1), as it's a direct formulaic calculation.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

// Unused or obfuscating macros and typedefs.
#define end return 0
typedef long long ll;
typedef long long int lli;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  // Fast I/O setup
  ios_base::sync_with_stdio(0);
  
  lli n, k, w, a, b, c; // WARNING: Confusing variable names.
  
  // Reads problem's k, n, w into variables n, k, w.
  cin >> n >> k >> w; 
  
  // `a` correctly calculates total cost: k * (w * (w+1) / 2)
  a = (w * (w + 1) / 2) * n;
  
  // `b` correctly calculates amount to borrow: TotalCost - InitialMoney
  b = a - k;
  
  if (a - k < 0) {
    cout << 0 << endl;
  } else {
    cout << b << endl;
  }
  return 0;
}
