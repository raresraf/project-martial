// https://codeforces.com/problemset/problem/158/A

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

using namespace std;

int main() {
  // your code goes here
  int n, m;
  cin >> n >> m;

  int k;
  vector<int> v;
  for (int i = 0; i < n; i++) {
    cin >> k;
    v.push_back(k);
  }
  sort(v.begin(), v.end());
  vector<int>::iterator it = upper_bound(v.begin(), v.end(), v[m]);
  cout << it - v.begin();
  // your code goes here
  return 0;
}