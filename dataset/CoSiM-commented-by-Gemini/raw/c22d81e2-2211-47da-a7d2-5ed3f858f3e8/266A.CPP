/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 266A - Stones on the Table
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  int n;
  string s;
  cin >> n >> s;

  int ans = 0;

  int i = 1;

  while (i < n) {
    while (s[i - 1] == s[i])
      ++i, ++ans;

    ++i;
  }

  cout << ans << "\n";

  return 0;
}