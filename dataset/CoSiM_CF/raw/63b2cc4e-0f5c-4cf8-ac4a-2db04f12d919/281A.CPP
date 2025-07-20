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
  if (str[0] >= 'A' && str[0] <= 'Z')
    ;
  else
    str[0] = str[0] + ('A' - 'a');
  cout << str << endl;
  return 0;
}
