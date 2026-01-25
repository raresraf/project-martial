#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
#define ten5 100000 + 10
#define MOD 1000000007
#define rep(i, a, n) for (int i = a; i < n; i++)
#define iif(c, t, f) ((c) ? (t) : (f))
#define per(i, a, n) for (int i = n - 1; i >= a; i--)
#define pb push_back
#define mp make_pair

int intcmp(const void *v1, const void *v2) { return *(int *)v1 - *(int *)v2; }

ll n, k;

int main() {
  cin >> n >> k;
  cout << (k / n) + !!(k % n) << endl;

  return 0;
}
