/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/69/A
 *
 * This program determines if a body is in equilibrium by checking if the net
 * force acting on it is zero. The net force is the vector sum of all individual
 * forces.
 */
#include <iostream>
using namespace std;
typedef long long ll;
int main() {
  // This array is declared but never used.
  int arr[101][101];
  int n;
  cin >> n;
  
  // a, b, and c will store the sum of the x, y, and z components of the forces.
  ll a = 0, b = 0, c = 0;
  ll a1, a2, a3;
  
  /**
   * @brief Read n force vectors and calculate the net force components.
   */
  for (int i = 0; i < n; i++) {
    cin >> a1 >> a2 >> a3;
    a += a1;
    b += a2;
    c += a3;
  }
  
  /**
   * @brief Check for equilibrium.
   * The body is in equilibrium if the net force along each axis is zero.
   */
  if (a == 0 && b == 0 && c == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
