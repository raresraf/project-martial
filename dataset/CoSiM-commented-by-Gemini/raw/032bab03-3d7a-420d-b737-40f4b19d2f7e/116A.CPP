/**
 * @file 116A.CPP
 * @brief Solution for a problem related to calculating minimum tram capacity.
 *        This appears to be from a competitive programming context, likely Codeforces "116A - Tram".
 *
 * @brief This program calculates the minimum required capacity for a tram based on the
 * number of passengers entering and exiting at each stop. The goal is to find the
 * maximum number of passengers inside the tram at any given time, which dictates
 * the minimum capacity needed.
 *
 * Algorithm:
 * - Initialize `ans` (current passengers) to 0 and `mx` (maximum passengers) to -1.
 * - Read the number of stops `t`.
 * - Iterate `t` times, once for each stop.
 * - In each iteration, read the number of exiting passengers (`a`) and entering
 *   passengers (`b`).
 * - Update the current number of passengers: `ans = ans - a + b`.
 * - Update the maximum number of passengers seen so far: `mx = max(mx, ans)`.
 * - After the loop, `mx` will hold the minimum capacity required.
 * - Print the result `mx`.
 *
 * Time Complexity: O(T), where T is the number of stops. The program iterates
 * through each stop exactly once.
 * Space Complexity: O(1), as the program uses a constant amount of extra space.
 */
//--------------**************---------------------
/*
 #        "    ""#    ""#                      #
 #   m  mmm      #      #     mmm    mmm    mmm#   mmm
 # m"     #      #      #    #"  "  #" "#  #" "#  #"  #
 #"#      #      #      #    #      #   #  #   #  #"""
 #  "m  mm#mm    "mm    "mm  "#mm"  "#m#"  "#m##  "#mm"  */

#include <bits/stdc++.h>
using namespace std;
#define pb push_back
#define mp make_pair
typedef pair<int, int> pii;
typedef long long ll;
typedef double ld;
typedef vector<int> vi;
#define fi first
#define se second
#define fe first
#define SZ 666666
#define si(n) scanf("%d", &n);
#define sl(n) scanf("%ld", &n);
#define pi(n) printf("%d\n", n);
#define pl(n) printf("%ld\n", n);
#define pf(n) printf("%f\n", n);
#define FILL(a, b) memset(a, 0, sizeof(b));
#define rep(i, n) for (int i = 0; i < n; i++)
#define reps(i, a, b) for (int i = 1; i <= b; i++)
const int INF = 1e9 + 5;
const int MOD = 1000000007;

//--------------**************---------------------

int main() {
  int t;
  si(t);
  int ans = 0;
  int mx = -1;
  /**
   * @brief Main loop to process each tram stop.
   * Invariant: `ans` holds the current number of passengers in the tram,
   * and `mx` holds the maximum number of passengers seen up to the current stop.
   */
  while (t--) {
    int a, b;
    // Read the number of exiting (a) and entering (b) passengers.
    cin >> a >> b;
    // Update the current number of passengers.
    ans = ans - a;
    ans = ans + b;
    // Update the maximum number of passengers if the current number is greater.
    mx = max(mx, ans);
  }
  // Output the minimum required capacity.
  cout << mx;
}