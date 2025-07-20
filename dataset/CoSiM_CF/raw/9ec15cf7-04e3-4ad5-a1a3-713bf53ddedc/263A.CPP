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
  int x, y, aux;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> aux;
      if (aux) {
        x = i;
        y = j;
      }
    }
  }
  cout << abs(x - 3) + abs(y - 3) << endl;
  return 0;
}
