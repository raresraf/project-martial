#include <bits/stdc++.h>
using namespace std;
#define endl "\n"
#define ll long long

int main() {
  ios_base ::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  ll n;
  cin >> n;
  ll sum = 0;
  for (ll i = 2; i < n; i++) {
    sum += i * (i + 1);
  }
  cout << sum << endl;
  return 0;
}