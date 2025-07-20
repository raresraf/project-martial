#include <bits/stdc++.h>

#define db double
#define ll long long
#define ar array
#define pb push_back
#define pop pop_back
#define endl "\n"
#define FOR(n) for (int i = 0; i < n; ++i)
#define FORL(s, e) for (int i = s; i < e; ++i)
#define FORM(s, e) for (int i = s; i <= e; ++i)
#define EACH(v) for (auto i : v)
//#define FORM(map, it) for(it = map.begin(); it != map.end(); ++it)
//#define unmap unordered_map<int, int>
//#define map map<int, int>
//#define sn set<int>

using namespace std;

const int mxN = 100;
string ans;

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  int n, cont1 = 0, cont2 = 0;
  cin >> n;
  int up[mxN];
  FOR(n) {
    cin >> up[i];
    if (up[i] == 100)
      cont1++;
    else
      cont2++;
  }
  if (cont1 % 2 != 0)
    cout << "NO" << endl;
  else if (cont2 % 2 != 0) {
    if (cont1 > 1)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  } else
    cout << "YES" << endl;
  return 0;
}
