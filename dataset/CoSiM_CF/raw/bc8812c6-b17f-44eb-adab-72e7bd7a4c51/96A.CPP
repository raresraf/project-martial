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
  string str;
  cin >> str;
  if (str.length() < 7) {
    cout << "NO";
    return 0;
  }
  for (int i = 0; i <= str.length() - 7; ++i) {
    if ((str[i] == '0') && (str[i + 1] == '0') && (str[i + 2] == '0') &&
        (str[i + 3] == '0') && (str[i + 4] == '0') && (str[i + 5] == '0') &&
        (str[i + 6] == '0')) {
      cout << "YES";
      return 0;
    } else if ((str[i] == '1') && (str[i + 1] == '1') && (str[i + 2] == '1') &&
               (str[i + 3] == '1') && (str[i + 4] == '1') &&
               (str[i + 5] == '1') && (str[i + 6] == '1')) {
      cout << "YES";
      return 0;
    }
  }
  cout << "NO";
  return 0;
}