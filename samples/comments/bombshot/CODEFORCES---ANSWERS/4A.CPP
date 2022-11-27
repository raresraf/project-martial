// headerfile//////////////
#include <bits/stdc++.h>
// define///////////////////
#define S(x) scanf("%d", &x)
#define P(x) printf("%d\n", x)
#define szo(x) ((int)sizeof(x))
#define szs(x) ((int)(x).size())
#define S2(x, y) scanf("%d%d", &x, &y)
#define loop(i, a, b) for (int i = a; i < b; i++)
#define revloop(i, a, n) for (int i = n - 1; i >= a; i--)
//////typedef//////////
typedef long long int ll;
typedef unsigned long long int ull;
const ll mod = 1000000007;
const int INF = 1000000000;
using namespace std;
//////functions////////
ull pow2(ull s) {
  if (s != 0 && (s & (s - 1)))
    cout << "0";
  else
    cout << "1";
}
ull powany(ll a, ll b) {
  ull res = 1;
  a %= mod;
  for (; b; b >>= 1) {
    if (b & 1)
      res = res * a % mod;
    a = a * a % mod;
  }
  return res;
}
ull prime(ull n) {
  if (n == 2 || n == 3)
    return 1;
  if (n % 2 == 0 || n % 3 == 0)
    return 0;
  for (ull i = 7; i * i < n; i += 6)
    if (n % (i) == 0 || n % (i - 2) == 0)
      return 0;
  return 1;
}
// head//////////////////
int main() {
  int n;
  cin >> n;
  if (n <= 3) {
    cout << "NO";
  } else {
    if (n % 2 == 0) {
      cout << "YES";
    } else {
      cout << "NO";
    }
  }

  return 0;
}