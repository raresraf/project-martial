/**
 * @file 231A.CPP
 * @brief Logic for consensus-based decision filtering in a multi-agent environment.
 * 
 * Logic: Calculates the number of problems a team of three agents will attempt. 
 * A problem is selected only if a majority (at least 2 out of 3) of the agents 
 * express confidence (state = 1) in the solution.
 * 
 * Algorithm: Majority voting threshold (sum > 1).
 * Time Complexity: $O(N)$ where N is the number of problems.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <bits/stdc++.h>
using namespace std;

// Functional Utility: Standardized input macros for various numeric types.
#define fi(a) scanf("%d", &a);
#define fli(a) scanf("%ld", &a);
#define fll(a) scanf("%lld", &a);
#define pi(a) printf("%d\n", a);
#define ppi(i, a) printf("Case %d: %d\n", i, a);
#define ll long long

/**
 * Functional Utility: Optimizes stream I/O throughput by decoupling C++ and C standard streams.
 */
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);

/**
 * @brief Logic: Configures local file redirection for development environments.
 */
void FI() {
#ifndef ONLINE_JUDGE
  freopen("C:\\Users\\SHIHAB\\Desktop\\input.in", "r", stdin);
#endif // ONLINE_JUDGE
}

int main() {
  fast FI(); 
  
  int n, a, b, c, cnt = 0;
  // Reads the total number of problems to be evaluated.
  if (!(cin >> n)) return 0;

  /**
   * Block Logic: Iteratively evaluates problem solvability based on agent consensus.
   * Invariant: A problem contributes to the final count only if Confidence(Sum) >= 2.
   */
  while (n--) {
    // Reads binary confidence states for agents Petya, Vasya, and Tonya.
    cin >> a >> b >> c;
    
    /**
     * Functional Utility: Majority rule implementation.
     * Logic: Sum of states {0, 1} > 1 is mathematically equivalent to reaching 
     * the threshold of 2 confident agents.
     */
    if (a + b + c > 1)
      cnt++;
  }
  
  // Output the total number of problems where a consensus was reached.
  cout << cnt << endl;
  
  return 0;
}
