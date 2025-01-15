#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  unordered_map<char, int> mp;
  for (int i = 0; i < s.size(); i++) {
    mp[s[i]]++;
  }
  int n = mp.size();
  if (n % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }
  return 0;
}
