/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 69A - Young Physicist
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  int n;
  cin >> n;

  vector<int> ans(3);

  for (int i = 0; i < n; ++i) {
    int x, y, z;
    cin >> x >> y >> z;

    ans[0] += x;
    ans[1] += y;
    ans[2] += z;
  }

  if (count(begin(ans), end(ans), 0) == 3)
    cout << "YES\n";
  else
    cout << "NO\n";

  return 0;
}