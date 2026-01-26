#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a[3], c = 0, s = 0, i;
  cin >> n;
  while (n--) {
    c = 0;
    for (i = 0; i < 3; i++) {
      cin >> a[i];
      if (a[i] == 1) {
        c++;
      }
      // cout<<n<<" "<<i<<endl;
    }
    if (c >= 2) {
      s++;
    }
    // cout<<c<<endl;
  }
  cout << s << endl;
}
