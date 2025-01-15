#include <bits/stdc++.h>

#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i)
#define REP(i, n) FOR(i, 0, n)
#define all(a) a.begin(), a.end()
#define pb push_back

typedef unsigned long long llu;
typedef long long ll;
typedef long double ld;

using namespace std;

int main() {
  int n;
  int v[3] = {0};
  cin >> n;
  REP(i, n) {
    int a, b, c;
    cin >> a >> b >> c;
    v[0] += a;
    v[1] += b;
    v[2] += c;
  }
  if (v[0] == v[1] && v[0] == v[2] && v[0] == 0)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
