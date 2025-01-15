/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 266B - Queue at the School
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  int n, t;
  string s;
  cin >> n >> t >> s;

  while (t--) {
    int i = 0;

    while (i < n - 1) {
      if (s[i] == 'B' and s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i += 2;
      } else
        ++i;
    }
  }

  cout << s << "\n";

  return 0;
}