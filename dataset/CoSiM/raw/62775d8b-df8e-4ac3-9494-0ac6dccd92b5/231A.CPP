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
  int n, cnt = 0, aux;
  scanf("%d", &n);
  REP(i, n) {
    int inner = 0;
    REP(i, 3) {
      scanf("%d", &aux);
      inner += aux;
    }
    if (inner >= 2)
      cnt++;
  }
  cout << cnt << endl;
  return 0;
}
