/**
 * I'm sorry I was an idiot
 */
#include <iostream>

using namespace std;

int main() {
  int many;
  cin >> many;
  int xt = 0, yt = 0, zt = 0;
  while (many--) {
    int x, y, z;
    cin >> x >> y >> z;
    xt += x;
    yt += y;
    zt += z;
  }
  if (xt == yt && xt == zt && xt == 0) {
    cout << "YES" << endl;
    return 0;
  }
  cout << "NO" << endl;
  return 0;
}