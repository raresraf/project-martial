// https://codeforces.com/problemset/problem/71/A

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
  int n, diff = 0;
  cin >> n;
  string s;
  vector<char> c;
  vector<string> v;

  for (int i = 0; i < n; i++) {
    cin >> s;
    v.push_back(s);
  }

  for (int i = 0; i < n; i++) {

    if (v[i].length() > 10) {
      diff = v[i].length() - 2;
      copy(v[i].begin(), v[i].end(), back_inserter(c));
      cout << v[i].front() << diff << v[i].back() << "\n";
    }

    else
      cout << v[i] << "\n";
  }
  // your code goes here
  return 0;
}