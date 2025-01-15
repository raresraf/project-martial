#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  string s;
  cin >> s;
  set<char> st;
  for (int i = 0; i < s.size(); i++) {
    st.insert(s[i]);
  }
  if (st.size() % 2 == 0)
    cout << "CHAT WITH HER!\n";
  else
    cout << "IGNORE HIM!\n";

  return 0;
}
