#include <bits/stdc++.h>
#include <vector>
using namespace std;
int main() {
  vector<int> vect;
  int x, y, z, i, n;
  cin >> n;
  while (n--) {
    cin >> x;
    vect.push_back(x);
    cin >> y;
    vect.push_back(y);
    cin >> z;
    vect.push_back(z);
  }
  x = y = z = 0;
  for (i = 0; i < vect.size(); i += 3) {
    x = x + vect[i];
    y = y + vect[i + 1];
    z = z + vect[i + 2];
  }
  if (x == 0 && y == 0 && z == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
