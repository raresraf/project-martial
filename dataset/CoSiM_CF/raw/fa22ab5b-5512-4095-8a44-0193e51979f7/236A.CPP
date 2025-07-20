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
  set<char> sc;
  getline(cin, line);
  REP(i, line.size()) sc.insert(line[i]);
  printf("%s", sc.size() & 1 ? "IGNORE HIM!\n" : "CHAT WITH HER!\n");
  return 0;
}
