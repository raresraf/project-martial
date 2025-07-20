#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  vector<int> v;
  for (int i = 0; i < str.length(); i++) {
    if (str[i] != '+') {
      v.push_back(str[i] - '0');
    }
  }
  sort(v.begin(), v.end());
  for (int i = 0; i < v.size(); i++) {
    if (i != v.size() - 1)
      cout << v.at(i) << '+';
    else
      cout << v.at(i) << endl;
  }
  return 0;
}