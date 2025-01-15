#include <bits/stdc++.h>
using namespace std;
#define F first
#define S second
#define mp make_pair
#define pb push_back
#define arsz (1 << 20)
typedef long long ll;

bool lucky(ll n) {
  if (n == 0)
    return false;
  bool luck = true;
  while (n != 0) {
    if ((n % 10) != 4 && (n % 10) != 7)
      luck = false;
    n = n / 10;
  }
  if (luck)
    return true;
  else
    return false;
}

int nol(ll num) {
  int count = 0;
  while (num != 0) {
    if (num % 10 == 4 || num % 10 == 7)
      ++count;
    num /= 10;
  }
  return count;
}

int main() {
  // freopen("input.txt","r", stdin);
  ios_base::sync_with_stdio(false);
  ll n;
  cin >> n;
  int numluck = nol(n);
  // cout<<numluck<<endl;
  if (lucky(numluck))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}