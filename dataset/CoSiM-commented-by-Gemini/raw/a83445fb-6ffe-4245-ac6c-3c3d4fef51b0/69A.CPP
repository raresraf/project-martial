#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
using namespace std;
int main() {
  int n;
  cin >> n;
  vector<int> v(3);
  vector<vector<int>> t;
  for (int i = 0; i < n; i++) {
    // v.clear();
    for (int j = 0; j < 3; j++) {
      cin >> v[j];
    }
    t.push_back(v);
  }
  // iterator<vector<int>>::it;
  int a = 0, b = 0, c = 0;
  for (int i = 0; i < t.size(); i++) {
    for (int j = 0; j < t[i].size(); j++) {
      // cout<<t[i][j]<<" ";
      if (j == 0)
        a += t[i][j];
      else if (j == 1)
        b += t[i][j];
      else
        c += t[i][j];
    }
    // cout<<"\n";
  }
  // cout<<a<<" "<<b<<" "<<c<<"\n";
  if (a == 0 && b == 0 && c == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}