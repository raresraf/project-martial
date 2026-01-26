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
  string line;
  getline(cin, line);
  int cnt = 0;
  REP(i, line.size()) {
    if (line[i] == '4' || line[i] == '7')
      cnt++;
  }
  printf("%s", cnt == 4 || cnt == 7 ? "YES\n" : "NO\n");
  return 0;
}
