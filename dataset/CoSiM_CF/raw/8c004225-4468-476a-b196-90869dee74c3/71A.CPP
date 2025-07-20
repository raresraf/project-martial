#include <bits/stdc++.h>
using namespace std;
typedef string str;
#define mx 10000

int main() {
  ios_base::sync_with_stdio(false);
  // cin.tie(NULL);
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    str st;
    cin >> st;
    if (st.size() <= 10)
      cout << st << "\n";
    if (st.size() > 10)
      cout << st[0] << (st.length() - 2) << st[st.length() - 1] << "\n";
  }
  return (0);
}
