// Codeforces
// 339A - Helpful maths

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
using namespace std;

int main() {
  string s;
  cin >> s;

  vector<char> v;
  for (unsigned int i = 0; i < s.size(); i += 2) {
    v.push_back(s[i]);
  }

  sort(v.begin(), v.end());

  cout << v[0];
  for (unsigned int i = 1; i < v.size(); i++) {
    cout << "+" << v[i];
  }

  return 0;
}
