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
  scanf("%d%*c", &n);
  while (n--) {
    string line;
    getline(cin, line);
    if (line.size() > 10)
      cout << line[0] << line.size() - 2 << line[line.size() - 1] << endl;
    else
      cout << line << endl;
  }
  return 0;
}
