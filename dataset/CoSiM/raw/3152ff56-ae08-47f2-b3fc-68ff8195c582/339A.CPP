// link : https://codeforces.com/problemset/problem/339/A

#include <bits/stdc++.h>

using namespace std;

string s;

vector<char> st;

main() {

  cin >> s;

  for (int i = 0; i < s.size(); i++) {

    if (s[i] != '+')
      st.push_back(s[i]);
  }

  sort(st.begin(), st.end());

  while (!st.empty()) {

    cout << *st.begin();

    st.erase(st.begin());

    if (!st.empty())
      cout << "+";
  }
}
