#include <iostream>
#include <unordered_set>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  string s;
  cin >> s;

  unordered_set<char> st;

  for (auto c : s) {
    st.insert(c);
  }

  if (st.size() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }

  return 0;
}
