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
  string a, b;
  getline(cin, a);
  getline(cin, b);
  REP(i, a.size()) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }
  if (a.compare(b))
    printf("%s", a.compare(b) < 0 ? "-1\n" : "1\n");
  else
    printf("0\n");
  return 0;
}
