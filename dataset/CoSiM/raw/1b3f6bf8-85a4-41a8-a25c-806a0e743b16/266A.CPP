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
  int n, count = 0;
  cin >> n;
  char ch;
  vector<char> v;
  for (int i = 0; i < n; ++i) {
    cin >> ch;
    v.push_back(ch);
  }
  for (int i = 0; i < n - 1; ++i) {
    if (v[i] == v[i + 1]) {
      count++;
    }
  }
  cout << count;
  return 0;
}