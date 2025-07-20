#include <bits/stdc++.h>
using namespace std;
int main() {
  vector<int> v;
  int n, l = 0, e = 0, i, sum = 0;
  cin >> n;
  while (n--) {
    cin >> l >> e;
    sum += e - l;
    v.push_back(sum);
    // cout<< sum << endl;
  }
  cout << *max_element(v.begin(), v.end()) << endl;
}
