#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  string s;
  cin >> s;
  vector<int> v;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] >= 49 and s[i] <= 57) {
      v.push_back(s[i] - 48);
    }
  }
  sort(v.begin(), v.end());
  for (int i = 0; i < v.size(); i++) {
    if (i == v.size() - 1) {
      cout << v[i];
    } else
      cout << v[i] << "+";
  }
}
