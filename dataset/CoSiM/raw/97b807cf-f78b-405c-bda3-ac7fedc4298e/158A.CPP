#include <algorithm>
#include <iostream>
#include <vector>

#define ll long long

using namespace std;

int main() {
  ll n, k;
  cin >> n >> k;
  vector<ll> a(n);
  for (int i = 0; i < n; i++)
    cin >> a[i];
  // sort(a.begin(),a.end());
  ll count = 0;
  for (int i = 0; i < n; i++)
    if (a[i] >= a[k - 1] && a[i] > 0)
      count++;
  cout << count << "\n";
  return 0;
}