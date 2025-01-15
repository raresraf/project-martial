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
  int idx = 0, i = 0;
  string hi = "hello", line;
  getline(cin, line);
  while (idx < hi.size() && i < line.size()) {
    if (hi[idx] == line[i])
      idx++;
    i++;
  }
  if (idx == hi.size())
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
