/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 116A - Tram
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  int n;
  cin >> n;

  int ans = 0;
  int curPeople = 0;

  for (int i = 0; i < n; ++i) {
    int a, b;
    cin >> a >> b;

    curPeople -= a;
    curPeople += b;

    ans = max(ans, curPeople);
  }

  cout << ans << "\n";

  return 0;
}