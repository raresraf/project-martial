#include <iostream>
using namespace std;
int main() {
  int n, a, b, res = 0, mxm = 0;
  cin >> n;
  while (n--) {
    cin >> a >> b;
    res += b - a;
    if (res > mxm) {
      mxm = res;
    }
  }
  cout << mxm << endl;
}
