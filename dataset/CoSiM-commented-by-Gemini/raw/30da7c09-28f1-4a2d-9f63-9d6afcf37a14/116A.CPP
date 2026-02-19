/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem by calculating the minimum required vehicle capacity.
 * @details This program simulates the number of passengers on a tram over a series of stops
 * to find the maximum number of passengers at any one time. This maximum value represents
 * the minimum capacity the tram must have. This version uses numerous macros common in
 * competitive programming for brevity.
 *
 * @algorithm The solution uses a greedy, single-pass simulation. It iterates through each stop,
 * updating the current number of passengers ('in') by subtracting those who exit ('a') and
 * adding those who enter ('b'). A running maximum ('count') is updated at each stop to
 * track the peak passenger load.
 *
 * @time_complexity O(N), where N is the number of stops, as the program iterates through each stop once.
 * @space_complexity O(1), as it uses a fixed number of variables for the simulation.
 */
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

// Type definitions and macros common in competitive programming.
typedef unsigned long long ULL;
typedef long long LL;

#define REP(i, n) FOR(i, 0, n)
#define FOR(i, a, b) for (int i = a; i < b; i++)
#define ROF(i, a, b) for (int i = a; i > b; i--)
// Macro for getting an integer input using scanf.
#define GI                                                                     \
  ({                                                                           \
    int t;                                                                     \
    scanf("%d", &t);                                                           \
    t;                                                                         \
  })
// Macro for getting a long long input using cin.
#define GL                                                                     \
  ({                                                                           \
    LL t;                                                                      \
    cin >> t;                                                                  \
    t;                                                                         \
  })
#define GD                                                                     \
  ({                                                                           \
    double t;                                                                  \
    scanf("%lf", &t);                                                          \
    t;                                                                         \
  })
#define pb push_back
#define mp make_pair
#define MOD 1000000007
#define INF (int)1e8
#define EPS 1e-9
#define TR(a, it)                                                              \
  for (typeof((a).begin()) it = (a).begin(); it != (a).end(); ++it)
int main() {
  // n: The number of stops.
  long long n = GL;
  // count: The maximum passenger count observed (the required capacity).
  LL count = 0;
  // in: The current number of passengers on the tram.
  LL in = 0;
  /**
   * Block Logic: This loop simulates the tram's journey across all 'n' stops.
   * Invariant: At the start of each iteration, `in` holds the passenger count from the
   *            previous stop, and `count` holds the maximum count observed so far.
   */
  FOR(i, 0, n) {
    int a = GI; // Passengers exiting.
    int b = GI; // Passengers entering.
    // Update the current passenger count.
    in = in - a + b;
    // Update the maximum capacity required.
    count = max(count, in);
  };
  cout << count;
  return 0;
}