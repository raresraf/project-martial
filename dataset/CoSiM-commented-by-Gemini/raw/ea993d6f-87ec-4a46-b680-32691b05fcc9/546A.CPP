/**
 * @file 546A.CPP
 * @brief Solves the "Soldier and Bananas" problem from Codeforces.
 *
 * @details
 * A soldier buys `w` bananas. The first costs `k`, the second `2k`, ..., the
 * `w`-th costs `w*k`. Given the soldier has `n` dollars, this program
 * calculates how much money must be borrowed.
 *
 * @algorithm This solution iteratively calculates the total cost. It loops from 1
 * to `w`, adding the cost of each banana (`k*i`) to a running total. After
 * the loop, it compares the total cost to the money the soldier has (`n`) and
 * prints the difference if it's positive, otherwise it prints 0.
 *
 * @time_complexity O(W), where W is the number of bananas, due to the loop.
 * @space_complexity O(1).
 */
///        From dust I came and to dust I shall return
/**
                    Asadullah Bin Rahman
                          CSE-17
    HAJEE MOHAMMAD DANESH SCIENCE AND TECHNOLOGY UNIVERSITY
*/
#include <bits/stdc++.h>
#define sf(f) scanf("%d", &f)
#define pf(f) printf("%d\n", f)
#define PI 3.1415926535897932385
#define size 3000000
#define mp make_pair
#define pb push_back
#define one(x) __builtin_popcount(x)     /// int
#define onel(x) __builtin_popcountl(x)   /// long int
#define onell(x) __builtin_popcountll(x) /// long long
using namespace std;
typedef long long ll;

/// int dx[]= {-1,-1,0,0,1,1};
/// int dy[]= {-1,0,-1,1,0,1};
/// int dx[]= {0,0,1,-1};                 /*4 side move*/
/// int dy[]= {-1,1,0,0};                 /*4 side move*/
/// int dx[]= {1,1,0,-1,-1,-1,0,1};       /*8 side move*/
/// int dy[]= {0,1,1,1,0,-1,-1,-1};       /*8 side move*/
/// int dx[]={1,1,2,2,-1,-1,-2,-2};       /*knight move*/
/// int dy[]={2,-2,1,-1,2,-2,1,-1};       /*knight move*/
/// Always Remember   'A'=65,'Z'=90 'a'=97 'z'=122 '0'=48

int main() {
  int i, k, n, w, cost = 0, pay;
  sf(k);
  sf(n);
  sf(w);

  /**
   * @block_logic Iteratively calculates the total cost of `w` bananas.
   * The cost of the i-th banana is `k * i`.
   */
  for (i = 1; i <= w; i++) {
    cost += k * i;
  }

  /**
   * @block_logic Checks if the soldier has enough money. If the cost is greater
   * than the money `n`, the deficit is calculated and printed. Otherwise, 0 is printed.
   */
  if (cost <= n)
    pf(0);
  else {
    pay = cost - n;
    pf(pay);
  }

  return 0;
}
