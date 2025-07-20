// link : https://codeforces.com/problemset/problem/236/A

#include <bits/stdc++.h>

using namespace std;

string s;

set<char> st;

main() {

  cin >> s;

  for (int i = 0; i < s.size(); i++)
    st.insert(s[i]);

  if (st.size() % 2 == 0)
    cout << "CHAT WITH HER!";

  else
    cout << "IGNORE HIM!";
}
