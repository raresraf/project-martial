#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string a, b;
  char t1, t2;
  int ans = 0;
  cin >> a;
  cin >> b;
  for (int i = 0; i < a.length(); i++) {
    t1 = tolower(a[i]);
    t2 = tolower(b[i]);
    if (t1 > t2) {
      ans = 1;
      break;
    } else if (t1 < t2) {
      ans = -1;
      break;
    } else
      continue;
  }
  cout << ans;
}