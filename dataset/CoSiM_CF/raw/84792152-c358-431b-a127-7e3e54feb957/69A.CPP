#include <iostream>
using namespace std;
int main() {
  int N, x, y, z, Tx = 0, Ty = 0, Tz = 0;
  cin >> N;
  for (int i = 0; i < N; i++) {
    cin >> x >> y >> z;
    Tx += x;
    Ty += y;
    Tz += z;
  }
  if (Tx == 0 && Ty == 0 && Tz == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
