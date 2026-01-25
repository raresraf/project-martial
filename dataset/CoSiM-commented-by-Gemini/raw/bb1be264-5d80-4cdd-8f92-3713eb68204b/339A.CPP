#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>
#define MX 100000
#define fs first
#define sec second
#define TC                                                                     \
  int T;                                                                       \
  cin >> T;                                                                    \
  while (T--)
#define loop(i, a, c) for (int i = 0; i < a; i++)
#define loop1(i, a, c) for (int i = 1; i <= a; i++)
#define pb(a) push_back(a)
#define all(a) (a).begin(), (a).end()
#define mp(x, y) make_pair((x), (y))
#define ll long long int
using namespace std;
int main() {
  string s;
  cin >> s;
  string p = "";
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+') {
      p += s[i];
    }
  }
  sort(p.begin(), p.end());
  for (int i = 0; i < p.size() - 1; i++) {
    cout << p[i] << "+";
  }
  cout << p[p.size() - 1];
}
