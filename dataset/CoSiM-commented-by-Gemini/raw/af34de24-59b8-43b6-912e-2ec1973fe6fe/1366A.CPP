/**
 * @file 1366A.CPP
 * @brief Solves a resource allocation problem.
 *
 * This program calculates the maximum number of items that can be crafted given
 * two types of resources, 'a' and 'b'. The crafting seems to require a total
 * of 3 resource units, with at least one of each type (e.g., 1 'a' and 2 'b',
 * or 2 'a' and 1 'b').
 */
#include <bits/stdc++.h>
#define ll long long
using namespace std;
int main() {
  ll a, b, t;
  cin >> t;
  while (t--) {
    cin >> a >> b;
    /**
     * @brief If either resource is zero, no items can be made.
     */
    if (a == 0 || b == 0)
      cout << "0" << endl;
    /**
     * @brief If one resource is more than double the other, the smaller
     * resource is the limiting factor. For example, if we have 2 'a's and 10 'b's,
     * we can only make 2 items (2 'a's and 4 'b's).
     */
    else if (2 * a < b)
      cout << a << endl;
    else if (2 * b < a)
      cout << b << endl;
    /**
     * @brief If the resources are relatively balanced, the total number of items
     * that can be made is the total number of resources divided by 3.
     */
    else
      cout << (a + b) / 3 << endl;
  }
}
