/**
 * @file 1426C.CPP
 * @brief Solution for Codeforces Problem 1426C - "Increase and Copy".
 * @details The problem asks for the minimum number of moves to obtain a sum of at least 'n'
 * starting from a single element with value 1. The allowed moves are:
 * 1. Increase an element's value by 1.
 * 2. Copy an element to create a new element.
 *
 * Algorithm:
 * Let's say we perform 'a' copy operations. This results in 'a + 1' elements in total.
 * To reach a sum of at least 'n' with the minimum number of 'increase' operations,
 * we should distribute the sum as evenly as possible. This means each of the 'a + 1'
 * elements should have a value of at least `ceil(n / (a + 1))`.
 * To raise the initial element from 1 to this value requires `ceil(n / (a + 1)) - 1` increase moves.
 * These increases must be done before the copies.
 *
 * The total number of moves for a chosen 'a' is the sum of copy moves and increase moves:
 *   Total Moves = a + (ceil(n / (a + 1)) - 1)
 *
 * The goal is to find an integer 'a' that minimizes this function. The function is convex,
 * and its minimum occurs when `a + 1` is close to `sqrt(n)`.
 * The code finds the optimal number of copy moves, `a`, which is `ceil(sqrt(n)) - 1`,
 * and then calculates the minimum total moves using the formula.
 *
 * Time Complexity: O(log N) for the binary search per test case.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

#define ll long long int
#define mp make_pair
#define pb push_back
#define fi first
#define si second
#define fastIO                                                                 
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL);
#define F(a, n)                                                                
  for (int i = 0; i < n; i++) {                                                
    cin >> a[i];                                                               
  }
#define F1(a, n)                                                               
  for (int i = 1; i <= n; i++) {                                               
    cin >> a[i];                                                               
  }
#define P(a, n)                                                                
  for (int i = 0; i < n; i++) {                                                
    cout << a[i] << ' ';                                                       
  }                                                                            
  cout << endl;
#define P1(a, n)                                                               
  for (int i = 1; i <= n; i++) {                                               
    cout << a[i] << ' ';                                                       
  }                                                                            
  cout << endl;
#define NF(a, n, m)                                                            
  for (int i = 0; i < n; i++) {                                                
    for (int j = 0; j < m; j++) {                                              
      cin >> a[i][j];                                                          
    }                                                                          
  }
#define NF1(a, n, m)                                                           
  for (int i = 1; i <= n; i++) {                                               
    for (int j = 1; j <= m; j++) {                                             
      cin >> a[i][j];                                                          
    }                                                                          
  }
#define PNF(a, n, m)                                                           
  for (int i = 0; i < n; i++) {                                                
    for (int j = 0; j < m; j++) {                                              
      cout << a[i][j] << ' ';                                                  
    }                                                                          
    cout << endl;                                                              
  }                                                                            
  cout << endl;
#define PNF1(a, n, m)                                                          
  for (int i = 1; i <= n; i++) {                                               
    for (int j = 1; j <= m; j++) {                                             
      cout << a[i][j] << ' ';                                                  
    }                                                                          
    cout << endl;                                                              
  }                                                                            
  cout << endl;
#define ceil_div(x, y) (((x) + (y)-1) / (y))
const int nax = 1e7;
const int mod = 1e9 + 7;
ll n;

// This function is unused in the final solution but demonstrates the formula.
ll tryit(ll a) {
  if (a >= n) {
    return a;
  }
  ll b = ceil_div(n, a + 1) - 1;
  return a + b;
}

/**
 * @brief Solves a single test case for a given 'n'.
 * @details This function determines the minimum moves required. It uses a binary search
 * to find the optimal number of copy moves ('a' or 'mid' in the code), which is known
 * to be near sqrt(n). The search efficiently finds the smallest 'mid' such that (mid+1)^2 >= n,
 * which corresponds to the optimal number of copy moves `a = ceil(sqrt(n)) - 1`.
 * It then calculates the total moves using this optimal 'a'.
 */
void func() {
  cin >> n;
  ll s = 0, e = 1e10, ans;
  /**
   * Block Logic: Binary search to find the optimal number of copy moves ('mid').
   * The search finds the smallest 'mid' where (mid+1)^2 is at least 'n'. This 'mid'
   * represents the number of copy operations 'a' that minimizes the total moves function.
   */
  while (s <= e) {
    ll mid = (s + e) / 2;
    ll val = (1 + mid) * (1 + mid);
    if (val >= n) {
      // If (mid+1)^2 >= n, we have a candidate for the optimal number of copy moves.
      // We calculate the total moves for this 'mid'.
      // The `mid >= n` check is a heuristic for corner cases where n is very small.
      ans = mid >= n ? mid : mid + ceil_div(n, mid + 1) - 1;
      // Try to find an even smaller 'mid' that still satisfies the condition.
      e = mid - 1;
    } else {
      // If (mid+1)^2 < n, 'mid' is too small. We need more copy moves.
      s = mid + 1;
    }
  }
  cout << ans << endl;
}

/**
 * @brief Main function to handle test cases.
 */
int main() {
  // Functional Utility: Fast I/O for competitive programming.
  fastIO
#ifndef ONLINE_JUDGE
      // For local testing: redirect input and output to files.
      freopen("../inp.txt", "r", stdin);
  freopen("../out.txt", "w", stdout);
#endif
  int t;
  cin >> t;
  // Invariant: Loop 't' times to solve each test case.
  while (t--) {
    func();
  }
}
