#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  ll t;
  cin >> t;
  while (t--) {
    ll n;
    cin >> n;
    int ar[n];
    for (int i = 0; i < n; i++)
      cin >> ar[i];
    vector<int> ans;
    ans.push_back(ar[0]);
    for (int i = 1; i < n; i++) {
      int tmp = ans[ans.size() - 1];
      if (tmp < 0 && ar[i] < 0)
        ans[ans.size() - 1] = max(tmp, ar[i]);
      if (tmp > 0 && ar[i] > 0)
        ans[ans.size() - 1] = max(tmp, ar[i]);
      if (tmp < 0 && ar[i] > 0)
        ans.push_back(ar[i]);
      if (tmp > 0 && ar[i] < 0)
        ans.push_back(ar[i]);
    }
    ll sans = 0;
    for (auto i : ans)
      sans += i;
    cout << sans << "\n";
  }
}
