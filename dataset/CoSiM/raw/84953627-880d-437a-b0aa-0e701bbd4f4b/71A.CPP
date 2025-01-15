#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    int sz = s.size();
    if (sz <= 10) {
      cout << s << endl;
      continue;
    }
    cout << s.front() << sz - 2 << s.back() << endl;
  }
  return 0;
}
