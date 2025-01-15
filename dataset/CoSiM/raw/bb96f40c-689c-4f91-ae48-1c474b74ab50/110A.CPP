#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  string n;
  cin >> n;
  int ret = 0;
  for (char ch : n)
    if (ch == '4' || ch == '7')
      ret++;
  cout << ((ret == 4 || ret == 7) ? "YES" : "NO") << endl;
  return 0;
}
