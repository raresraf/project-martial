#include <iostream>
#include <set>

using namespace std;

int main() {
  string s;
  cin >> s;
  int sz = s.size();
  set<char> st;
  for (int i = 0; i < sz; ++i) {
    st.insert(s[i]);
  }
  int ssz = st.size();

  if (ssz % 2 != 0)
    cout << "IGNORE HIM!\n";
  else
    cout << "CHAT WITH HER!\n";

  return 0;
}
