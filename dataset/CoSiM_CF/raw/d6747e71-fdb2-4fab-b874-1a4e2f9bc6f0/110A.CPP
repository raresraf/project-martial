/**
* Git >> Mohsin_Riad


















**/
#include <bits/stdc++.h>
using namespace std;

#define IOS                                                                    \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define mx 1000
#define endl "\n"
#define pb push_back
#define pob pop_back
typedef long long ll;
typedef double dd;
const int N = 1e6;

ll n;

int calc(ll x) {
  ll cnt1 = 0, cnt2 = 0, cnt3 = 0;
  while (x != 0) {
    cnt1++;
    ll temp = x % 10;
    if (temp == 4)
      cnt2++;
    if (temp == 7)
      cnt3++;
    x /= 10;
  }
  if (cnt2 == 0 || cnt3 == 0)
    return 0;
  else {
    if (cnt1 == (cnt2 + cnt3))
      return 1;
    else
      return 0;
  }
}

int main() {
  IOS

          cin >>
      n;
  bool luck = calc(n);
  if (luck == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return (0);
}
