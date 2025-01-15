#include <bits/stdc++.h>
#define ll long long

using namespace std;
int main() {
  ll n;
  cin >> n;
  ll arr[n];

  for (ll i = 0; i < n; i++) {
    cin >> arr[i];
  }

  ll s, k;

  cin >> s;
  cin >> k;
  ll z;
  z = s * k;
  int b;
  if (k == 0)

  {
    b = s;
  } else {
    b = arr[z - 1];
    // cout<<arr[z-1];
  }

  cout << b;
  return 0;
}