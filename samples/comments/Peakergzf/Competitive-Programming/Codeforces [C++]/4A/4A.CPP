#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pii;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  int w;
  cin >> w;
  string ans = "No";
  if (w > 2 && w % 2 == 0)
    ans = "YES";
  cout << ans << endl;
  return 0;
}
