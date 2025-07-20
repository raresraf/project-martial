#include <algorithm>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <vector>
#define int long long
#define INF 10000000000000000
#define mir(i) (n - 1 - i)
#define mid(l, u) ((l + u) / 2)
#define lchild(i) (i * 2 + 1)
#define rchild(i) (i * 2 + 2)
using namespace std;
int32_t main() {
  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    if (s.length() <= 10) {
      cout << s << endl;
      continue;
    }
    cout << s[0] << (s.length() - 2) << s[s.length() - 1] << endl;
  }
}