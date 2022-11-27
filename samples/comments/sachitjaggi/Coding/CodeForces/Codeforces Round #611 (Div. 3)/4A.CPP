#include <bits/stdc++.h>
using namespace std;

int main() {
  // #ifndef ONLINE_JUDGE
  // freopen("input.txt","r",stdin);
  // freopen("output.txt","w",stdout);
  // #endif
  long long n, m;
  cin >> n >> m;
  queue<pair<long long, long long>> curr;
  map<long long, bool> mp;
  mp.clear();
  for (long long i = 0; i < n; i++) {
    long long temp, lev = 0;
    cin >> temp;
    curr.push({temp, lev});
    mp[temp] = true;
  }
  vector<long long> values;
  long long ans = 0, left = m;
  while (left != 0) {
    pair<long long, long long> temp = curr.front();
    curr.pop();
    map<long long, bool>::iterator it_left, it_right;
    it_left = mp.find(temp.first - 1);
    it_right = mp.find(temp.first + 1);
    if (it_left == mp.end()) {
      values.push_back(temp.first - 1);
      left--;
      curr.push({temp.first - 1, temp.second + 1});
      ans += temp.second + 1;
      mp[temp.first - 1] = true;
    }
    if (left == 0) {
      break;
    }
    if (it_right == mp.end()) {
      mp[temp.first + 1] = true;

      values.push_back(temp.first + 1);
      left--;
      curr.push({temp.first + 1, temp.second + 1});
      ans += temp.second + 1;
    }
  }
  curr.empty();
  cout << ans << endl;
  for (long long i = 0; i < m; i++) {
    cout << values[i] << " ";
  }
}