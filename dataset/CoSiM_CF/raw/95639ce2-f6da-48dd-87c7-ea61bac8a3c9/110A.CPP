/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 110A - Nearly Lucky Number
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  string s;
  cin >> s;

  int amount = 0;
  string ans = "YES";

  for (char &c : s) {
    if (c == '4' or c == '7')
      ++amount;
  }

  if (amount == 0)
    ans = "NO";

  while (amount > 0) {
    int d = amount % 10;

    if (d != 4 and d != 7) {
      ans = "NO";
      break;
    }

    amount /= 10;
  }

  cout << ans << "\n";

  return 0;
}
