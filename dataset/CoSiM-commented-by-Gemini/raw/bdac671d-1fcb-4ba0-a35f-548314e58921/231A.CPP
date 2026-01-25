#include <iostream>

using namespace std;

int main() {
  int n, k, m, in, s = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> k >> m >> in;
    if (k + m + in > 1) {
      s++;
    }
  }
  cout << s << endl;
  return 0;
}
