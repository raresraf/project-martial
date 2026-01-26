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
  vector<int> num;
  for (int i = 0; i < line.size(); i += 2)
    num.pb(line[i] - '0');
  sort(all(num));
  REP(i, num.size()) printf("%d%c", num[i], i == num.size() - 1 ? '\n' : '+');
  return 0;
}
