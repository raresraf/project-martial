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
  bool dang = false;
  string line;
  getline(cin, line);
  int v[line.size()];
  REP(i, line.size()) v[i] = line[i] == '0' ? -1 : 1;
  REP(i, line.size()) if (i) v[i] += v[i - 1] * v[i] < 0 ? 0 : v[i - 1];
  REP(i, line.size()) if (abs(v[i]) >= 7) dang = true;
  printf("%s", dang ? "YES\n" : "NO\n");
  return 0;
}
