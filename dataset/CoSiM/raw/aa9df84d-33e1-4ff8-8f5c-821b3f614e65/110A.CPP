#include <bits/stdc++.h>
using namespace std;
#define Test(x)                                                                \
  int x;                                                                       \
  cin >> x
#define sz(x) ((int)x.size())
#define sqr(x) ((x) * (x))
#define pb push_back
#define mp make_pair

int main() {
  // ios_base::sync_with_stdio(0);
  string str;
  cin >> str;
  int lucky = 0;
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == '7' || str[i] == '4')
      lucky++;
  }
  if (lucky == 0) {
    cout << "NO\n";
    return 0;
  }
  while (lucky) {
    if ((lucky % 10) != 7 && (lucky % 10) != 4) {
      cout << "NO\n";
      return 0;
    }
    lucky /= 10;
  }
  cout << "YES\n";
  return 0;
}
