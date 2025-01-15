#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string a, b;
  cin >> a >> b;
  transform(a.begin(), a.end(), a.begin(), ::toupper);
  transform(b.begin(), b.end(), b.begin(), ::toupper);

  int x = 0;
  for (int i = 0; i < a.length(); i++) {
    if ((int)a[i] > (int)b[i]) {
      x = 1;
      break;
    } else if ((int)a[i] < (int)b[i]) {
      x = -1;
      break;
    }
  }
  cout << x << endl;
}