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
  string str1, str2;
  cin >> str1 >> str2;
  transform(str1.begin(), str1.end(), str1.begin(), ::tolower);
  transform(str2.begin(), str2.end(), str2.begin(), ::tolower);
  if (str1.compare(str2) == 0) {
    cout << "0";
    return 0;
  }
  for (int i = 0; i < str1.length(); ++i) {
    if ((int)str1[i] < (int)str2[i]) {
      cout << "-1";
      return 0;
    } else if (((int)str1[i] > (int)str2[i])) {
      cout << "1";
      return 0;
    }
  }

  return 0;
}