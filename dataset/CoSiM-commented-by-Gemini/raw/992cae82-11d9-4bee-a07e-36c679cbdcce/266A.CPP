/**
 * @file 266A.CPP
 * @author Jos Luiz da Silva Neto
 * @brief Solution for a competitive programming problem (Codeforces "Stones on the Table").
 * This program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color.
 *
 * NOTE: This file contains a large number of pre-defined macros and boilerplate
 * code common in competitive programming, most of which is not used in this solution.
 *
 * @algorithm Linear Scan and Counting. The algorithm iterates through the stones
 * once, comparing each stone with the one next to it.
 *
 * @time_complexity O(N), where N is the number of stones, due to the single loop.
 * @space_complexity O(N) to store the string representing the stone colors.
 */
/// Jos  Luiz da Silva Neto
/// Computer Engineering - Federal University of Itajub
#include <bits/stdc++.h>

using namespace std;

// --- Boilerplate competitive programming macros and typedefs ---
#define INF 0x3F3F3F3F
#define LINF 0x3F3F3F3F3F3F3F3FLL
#define pb push_back
#define mp make_pair
#define F first
#define S second

typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vii;
typedef long long ll;
typedef unsigned long long ull;

#define D(x) cout << #x << " = " << x << endl;

int vx[] = {1, 0, -1, 0};
int vy[] = {0, 1, 0, -1};
// --- End of Boilerplate ---

int cnt; // Global counter for stones to be removed.

int main() {
  int n;
  cin >> n;
  string x;
  cin >> x;

  /**
   * @block
   * @brief Counts the number of adjacent identical stones.
   *
   * The loop iterates up to the second-to-last stone and compares each stone
   * `x[i]` with its immediate neighbor `x[i+1]`. Each time a pair of identical
   * adjacent stones is found, it signifies that one must be removed.
   */
  for (int i = 0; i < n - 1; i++)
    if (x[i] == x[i + 1])
      cnt++;

  printf("%d\n", cnt);
  return 0;
}