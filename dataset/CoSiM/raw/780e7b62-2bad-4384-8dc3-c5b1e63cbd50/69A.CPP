#include <iostream>
using namespace std;
int main() {

  int x[101], y[101], z[101];
  int n;
  int xsum = 0, ysum = 0, zsum = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
    xsum += x[i];
    ysum += y[i];
    zsum += z[i];
  }
  if (xsum == 0 && ysum == 0 && zsum == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
