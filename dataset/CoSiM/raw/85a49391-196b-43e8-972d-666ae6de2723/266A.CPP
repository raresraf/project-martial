#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  int n;
  string s;
  cin >> n >> s;
  n++;
  s += '@';
  int ret = 0, cur = 1;
  char ch = s[0];
  for (int i = 1; i < n; i++)
    if (ch == s[i])
      cur++;
    else
      ch = s[i], ret += cur - 1, cur = 1;
  cout << ret << endl;
  return 0;
}
