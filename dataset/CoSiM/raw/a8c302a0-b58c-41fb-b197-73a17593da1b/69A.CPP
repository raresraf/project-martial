#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, v[] = {0, 0, 0};
  int a, b, c;

  cin >> n;

  for (int i = 0; i < n; ++i) {
    cin >> a >> b >> c;

    v[0] += a;
    v[1] += b;
    v[2] += c;
  }

  if (v[0] == 0 && v[1] == 0 && v[2] == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}