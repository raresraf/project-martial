#include <iostream>
#include <vector>
using namespace std;
int main() {
  typedef long long ll;
  ll n, k, m, t;
  cin >> n >> k;
  k--;
  vector<ll> v;
  for (ll i = 0; i < n; i++) {
    cin >> t;
    if (k == i)
      m = t;
    v.push_back(t);
  }
  ll count = 0;
  for (auto i = v.begin(); i < v.end(); i++) {
    if (*i >= m && *i != 0) {
      count++;
    }
  }
  cout << count << "\n";
}