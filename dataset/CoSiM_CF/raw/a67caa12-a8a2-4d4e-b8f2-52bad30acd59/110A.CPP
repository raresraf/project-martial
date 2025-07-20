#include <bits/stdc++.h>
using namespace std;
map<char, long long> m, m2;
int main() {
  ios_base::sync_with_stdio(0);
  string s;
  cin >> s;
  long long size = s.length();
  for (int i = 0; i < size; i++) {
    m[s[i]]++;
  }
  long long luck = m['7'] + m['4'];
  stringstream ss;
  string sss;
  ss << luck;
  ss >> sss;
  long long lucky = sss.length();
  for (int i = 0; i < lucky; i++) {
    m2[sss[i]]++;
  }
  long long cek = m2['7'] + m2['4'];
  if (cek == lucky) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
}
