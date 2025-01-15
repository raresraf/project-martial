/* ===================================*/
/* Author :   Moshiur Rahman Mohim    */
/* Email  :   mtmohim74@gmail.com     */
/* Institute: HSTU                    */
/* ===================================*/

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <algorithm>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
using namespace std;

#define pi acos(-1.0)
#define mod 1000000007
#define MAX 10000
#define FOR(i, b, n) for (LL i = b; i <= n; i++)
#define FORR(i, n, b) for (LL i = n; i >= b; i--)
// Library.
#define mem(a, b) memset(a, b, sizeof(a))
#define Sort(x) sort(x.begin(), x.end())
#define Reverse(x) reverse(x.begin(), x.end())
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define sq(x) (x) * (x)
// File.
#define READ(f) freopen(f, "r", stdin)
#define WRITE(f) freopen(f, "w", stdout)
// i/o.
#define sf scanf
#define pf printf
#define pr1(x) cout << x << "\n"
#define pr2(x, y) cout << x << " " << y << "\n"
#define pr3(x, y, z) cout << x << " " << y << " " << z << "\n"
// Test Case.
#define TEST(i, t)                                                             \
  LL i, t;                                                                     \
  scanf("%I64d", &t);                                                          \
  for (i = 1; i <= t; i++)

typedef unsigned int U;
typedef long int L;
typedef unsigned long int LU;
typedef long long int LL;
typedef unsigned long long int LLU;
typedef float F;
typedef double LF;
typedef char C;

// Compute b^p%m
int BigMod(LL B, LL P, LL M) {
  LL R = 1;
  while (P > 0) {
    if (P % 2 == 1)
      R = (R * B) % M;
    P /= 2;
    B = (B * B) % M;
  }
  return (int)R;
}

/*............End............*/
/*.......Global Declaration.......*/
C c;
int main() {
  // READ("input.txt");
  // WRITE("output.txt");
  LL ch[100], n;
  FOR(i, 0, 100) {
    sf("%I64d%c", &ch[i], &c);
    if (c == '\n') {
      n = i + 1;
      break;
    }
  }
  sort(ch, ch + n);
  FOR(i, 0, n - 1) {
    pf("%I64d", ch[i]);
    if (i != n - 1)
      pf("+");
  }
  return 0;
}
