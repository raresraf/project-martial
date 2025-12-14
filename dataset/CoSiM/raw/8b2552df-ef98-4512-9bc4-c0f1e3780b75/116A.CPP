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
  int curr = 0, n, m = 0;
  scanf("%d", &n);
  while (n--) {
    int a, b;
    scanf("%d %d", &a, &b);
    curr -= a;
    curr += b;
    m = max(curr, m);
  }
  cout << m << endl;
  return 0;
}
