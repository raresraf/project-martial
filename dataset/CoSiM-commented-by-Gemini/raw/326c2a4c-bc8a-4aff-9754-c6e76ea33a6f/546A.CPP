/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem.
 * @details This program calculates the amount of money a soldier needs to borrow to buy a
 * specific number of bananas, where the cost of each banana increases arithmetically.
 *
 * @algorithm The solution calculates the total cost of `w` bananas using the formula for the
 * sum of an arithmetic series: `k * (1 + 2 + ... + w)`. This is simplified to
 * `k * w * (w + 1) / 2`. The amount to borrow is then the total cost minus the soldier's
 * initial money `n`. The `max` function is used to ensure the result is at least 0.
 *
 * @time_complexity O(1) for each test case, as it uses a direct mathematical formula.
 * @space_complexity O(1).
 */
// Author: btjanaka (Bryon Tjanaka)
// Problem: (CodeForces) 546a
// Title: Soldier and Bananas
// Link: https://codeforces.com/problemset/problem/546/A
// Idea: Use the formula for sum of an arithmetic sequence.
// Difficulty: math
#include <bits/stdc++.h>
#define GET(x) scanf("%d", &x)
#define GED(x) scanf("%lf", &x)
using namespace std;
typedef long long ll;
typedef pair<int, int> ii;
typedef vector<int> vi;

int main() {
  int k, n, w;
  /**
   * Block Logic: This loop is structured to handle multiple test cases until End-of-File.
   * `scanf` returns the number of items read, which is > 0 on success.
   */
  while (GET(k) > 0) {
    GET(n);
    GET(w);
    // Block Logic: Calculate the total cost and the amount to borrow.
    // The total cost is calculated using the arithmetic series sum formula.
    // `long long` is used for `tot` to prevent potential integer overflow.
    ll tot = w * (w + 1) / 2 * k;
    // The amount to borrow is the deficit, or 0 if the soldier has enough money.
    printf("%lld\n", max(0ll, tot - n));
  }
  return 0;
}
