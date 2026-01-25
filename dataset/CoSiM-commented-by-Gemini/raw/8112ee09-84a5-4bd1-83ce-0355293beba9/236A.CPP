#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  set<char> st;
  int i, len = s.length();
  for (i = 0; i < len; i++) {
    st.insert(s[i]);
  }
  int a = st.size();
  if (a % 2 == 0)
    printf("CHAT WITH HER!\n");
  else
    printf("IGNORE HIM!\n");
  return 0;
}
