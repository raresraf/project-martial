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
  int n;
  cin >> n;
  int ans = 0;
  string temp;
  string str1 = "X++";
  string str2 = "++X";
  string str3 = "X--";
  string str4 = "--X";
  for (int i = 0; i < n; ++i) {
    cin >> temp;
    if ((temp.compare(str1) == 0) || (temp.compare(str2) == 0))
      ans++;
    else
      ans--;
  }
  cout << ans;
  return 0;
}