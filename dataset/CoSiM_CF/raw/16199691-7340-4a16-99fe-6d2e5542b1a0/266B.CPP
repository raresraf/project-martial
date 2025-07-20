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
  int n, t;
  scanf("%d %d%*c", &n, &t);
  string line;
  getline(cin, line);
  while (t) {
    REP(i, line.size()) {
      if (i != line.size() - 1 && line[i] == 'B' && line[i + 1] == 'G' && t) {
        swap(line[i], line[i + 1]);
        i++;
      }
    }
    t--;
  }
  cout << line << endl;
  return 0;
}
