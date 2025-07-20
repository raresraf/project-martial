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
  int n, cnt = 1;
  char curr;
  string line;
  scanf("%d%*c", &n);
  getline(cin, line);
  curr = line[0];
  FOR(i, 1, line.size()) {
    if (curr != line[i]) {
      curr = line[i];
      cnt++;
    }
  }
  cout << line.size() - cnt << endl;
  return 0;
}
