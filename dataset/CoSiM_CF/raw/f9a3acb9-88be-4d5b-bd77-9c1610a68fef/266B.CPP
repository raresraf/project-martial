// https://codeforces.com/problemset/problem/266/B

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
  int n, t;
  string s;
  cin >> n >> t >> s;
  while (t--) {
    for (int i = 1; i < n; ++i) {
      if (s[i] == 'G' && s[i - 1] == 'B') {
        s[i] = 'B';
        s[i - 1] = 'G';
        ++i;
      }
    }
  }
  cout << s << endl;
  // your code goes here
  return 0;
}