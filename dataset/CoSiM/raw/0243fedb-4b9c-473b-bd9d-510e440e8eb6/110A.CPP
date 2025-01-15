#include <bits/stdc++.h>
#define ll long long
using namespace std;
bool count(ll int);
int main() {
  ll a;

  cin >> a;
  bool c;
  c = count(a);

  if (c) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
bool count(ll int n) {
  int con = 0;
  while (n != 0) {
    if (n % 10 == 4 || n % 10 == 7) {
      con++;
    }
    n = n / 10;
  }
  if (con == 4 || con == 7) {
    return true;
  } else {
    return false;
  }
}
