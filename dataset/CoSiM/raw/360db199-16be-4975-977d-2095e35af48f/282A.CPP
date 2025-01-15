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
  int n, x = 0;
  scanf("%d%*c", &n);
  while (n--) {
    int p = 0, m = 0;
    string line;
    getline(cin, line);
    REP(i, line.size()) {
      if (line[i] == '+')
        p++;
      else if (line[i] == '-')
        m++;
    }
    if (p == 2)
      x++;
    else if (m == 2)
      x--;
  }
  cout << x << endl;
  return 0;
}
