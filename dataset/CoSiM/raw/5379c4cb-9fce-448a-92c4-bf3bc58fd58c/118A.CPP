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
  string line, res;
  getline(cin, line);
  REP(i, line.size()) {
    char aux = tolower(line[i]);
    if (!(aux == 'a' || aux == 'o' || aux == 'y' || aux == 'e' || aux == 'u' ||
          aux == 'i')) {
      res.pb('.');
      res.pb(aux);
    }
  }
  cout << res << endl;
  return 0;
}
