

#include <bits/stdc++.h>
typedef long long int ll;
#define elif else if
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define rep(i, a, b) for (ll i = a; i < b; i++)
using namespace std;
int main() {
  ll costofonebanana, initial_money, n;
  cin >> costofonebanana >> initial_money >> n;
  ll total_money = (n * (n + 1));
  total_money = total_money / 2;
  total_money = total_money * (costofonebanana);
  ll res = total_money - initial_money;
  if (res <= 0) {
    cout << 0 << endl;
  } else {
    cout << res << endl;
  }
  return 0;
}