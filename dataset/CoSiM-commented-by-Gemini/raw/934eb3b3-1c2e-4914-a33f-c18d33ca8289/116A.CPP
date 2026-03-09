/**
 * @file 116A.CPP
 * @brief Calculates the minimum capacity of a tram.
 * @details This program solves the "Tram" problem from Codeforces. It determines the
 *          minimum required capacity of a tram by tracking the number of passengers
 *          entering and exiting at each stop and finding the maximum number of
 *          passengers on board at any given time.
 *
 * @author Jos  Luiz da Silva Neto
 * @institution Computer Engineering - Federal University of Itajub
 *
 * @algorithm Simulation / Maximum Tracking
 * @complexity
 * - Time: O(N), where N is the number of stops, due to a single loop through the stops.
 * - Space: O(1), as the memory usage is constant regardless of the input size.
 */
#include <bits/stdc++.h>

using namespace std;

// Common competitive programming macros for brevity.
#define INF 0x3F3F3F3F
#define LINF 0x3F3F3F3F3F3F3F3FLL
#define pb push_back
#define mp make_pair
#define F first
#define S second

// Common competitive programming type definitions.
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vii;
typedef long long ll;
typedef unsigned long long ull;

// Macro for debugging purposes.
#define D(x) cout << #x << " = " << x << endl;

// Arrays for grid-based problems (e.g., BFS/DFS), not used in this solution.
int vx[] = {1, 0, -1, 0};
int vy[] = {0, 1, 0, -1};

int main() {
  // n: the number of tram stops.
  int n;
  // resp: tracks the maximum capacity needed (the answer).
  // tam: tracks the current number of passengers on the tram.
  int resp = 0, tam = 0;
  cin >> n;

  /**
   * @brief Block Logic: Simulates the passenger flow at each stop.
   * Invariant: After each iteration, `tam` holds the current number of passengers,
   *            and `resp` holds the maximum number of passengers seen up to that point.
   */
  for (int i = 0; i < n; i++) {
    // x: passengers exiting, y: passengers entering.
    int x, y;
    cin >> x >> y;
    // Update the current number of passengers.
    tam = tam - x + y;
    // Update the required capacity if the current occupancy is a new maximum.
    resp = max(resp, tam);
    /**
     * @note The following line `tam = max(tam, 0);` is logically incorrect for this
     * specific problem's constraints, as the number of exiting passengers never
     * exceeds the number on board. It seems to be a defensive check against an
     * invalid state that doesn't occur in the problem. It is left as is,
     * as per the original code.
     */
    tam = max(tam, 0);
  }
  printf("%d\n", resp);
  return 0;
}
