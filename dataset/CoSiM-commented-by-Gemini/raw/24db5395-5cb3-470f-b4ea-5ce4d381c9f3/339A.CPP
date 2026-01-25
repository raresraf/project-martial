#include <bits/stdc++.h>
#include <string.h>
using namespace std;
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"
#define Max(x, y, z) max(x, max(y, z))
#define Min(x, y, z) min(x, min(y, z))
#define int long long
#define trace1(x) cerr << #x << ": " << x << endl
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl
#define trace3(x, y, z)                                                        \
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << endl

int32_t main() {
  IOS;
  string s;
  vector<int> v;
  cin >> s;
  for (int i = 0; i < s.length(); ++i) {
    if (s[i] == '1')
      v.push_back(1);
    else if (s[i] == '2')
      v.push_back(2);
    else if (s[i] == '3')
      v.push_back(3);
  }
  sort(v.begin(), v.end());
  cout << v[0];
  for (int i = 1; i < v.size(); ++i) {
    cout << "+" << v[i];
  }
  return 0;
}